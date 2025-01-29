import os
import json
import random
import numpy as np
import torch
import requests # Import requests for API calls - still needed in process_completion_request
from colorama import init, Fore, Style
from configparser import ConfigParser

from ..utils.Cerebras_api_utils import load_prompt_options, get_prompt_content, fetch_cerebras_models
from ..utils.Cerebras_chat_utils import ChatHistoryManager

init()  # Initialize colorama

class CerebrasAPILLM:
    DEFAULT_PROMPT = "Use [system_message] and [user_input]"

    _LLM_MODELS = []  # Private class-level list to store models

    def __init__(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        cerebras_directory = os.path.join(current_directory, 'cerebras')
        config_path = os.path.join(cerebras_directory, 'CerebrasConfig.ini')
        self.config = ConfigParser()
        self.config.read(config_path)
        self.api_key = self.config.get('API', 'key') # Assuming API key is in config

        self.cerebras_api_base_url = "https://api.cerebras.ai/v1" # Base URL for Cerebras API

        # Fetch LLM models from Cerebras API during initialization and store in instance
        self.instance_llm_models = fetch_cerebras_models(self.api_key, self.cerebras_api_base_url)
        if not self.instance_llm_models:
            print(Fore.RED + "Failed to fetch Cerebras models from API. Check API key and connection." + Style.RESET_ALL)
            self.instance_llm_models = ["error_fetching_models"] # Fallback in case of error
        CerebrasAPILLM._LLM_MODELS = self.instance_llm_models # Update the class-level list too, for fallback maybe


        # Load prompt options
        prompt_files = [
            os.path.join(cerebras_directory, 'DefaultPrompts.json'),
            os.path.join(cerebras_directory, 'UserPrompts.json')
        ]
        self.prompt_options = load_prompt_options(prompt_files)

        # Initialize chat history manager
        self.chat_history_manager = ChatHistoryManager()

    @classmethod
    def LLM_MODELS(cls): # Class property to access the models
        instance = cls() # Create a temporary instance to access instance_llm_models
        return instance.instance_llm_models # Return the instance's model list

    @classmethod
    def INPUT_TYPES(cls):
        # Use the LLM_MODELS class property to get the list for the dropdown
        model_choices = cls.LLM_MODELS() # Call the class property as a method

        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            cerebras_directory = os.path.join(current_directory, 'cerebras')
            prompt_files = [
                os.path.join(cerebras_directory, 'DefaultPrompts.json'),
                os.path.join(cerebras_directory, 'UserPrompts.json')
            ]
            prompt_options = load_prompt_options(prompt_files)
        except Exception as e:
            print(Fore.RED + f"Failed to load prompt options: {e}" + Style.RESET_ALL)
            prompt_options = {}

        return {
            "required": {
                "model": (model_choices, {"tooltip": "Select the Cerebras LLM model to use.", "type": "COMBO"}), # Use COMBO and model_choices
                "preset": ([cls.DEFAULT_PROMPT] + list(prompt_options.keys()), {"tooltip": "Select a preset or custom prompt for guiding the LLM."}),
                "system_message": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system message to guide the LLM's behavior."}),
                "user_input": ("STRING", {"multiline": True, "default": "", "tooltip": "User input or prompt to generate a response from the LLM."}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Controls randomness in responses. 0.0 is deterministic."}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072, "step": 1, "tooltip": "Maximum number of tokens to generate in the response."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Limits the pool of words the model can choose from based on probability mass. 1.0 to consider all tokens."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295, "tooltip": "Seed for random number generation, ensuring reproducibility."}),
                "stop": ("STRING", {"default": "", "tooltip": "Stop generation when the specified sequence is encountered."}),
                "json_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable JSON mode for structured output if supported by API and model."}), # Tooltip updated
                "conversation_id": ("STRING", {"default": "", "tooltip": "Unique identifier for the conversation. Leave empty for a new conversation."}),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("api_response", "success", "conversation_id", "chat_history")
    OUTPUT_TOOLTIPS = ("The API response (generated text).", "Whether the request was successful.", "The unique identifier for the conversation.", "The complete chat history (JSON string).")
    FUNCTION = "process_completion_request"
    CATEGORY = "⚡ MNeMiC Nodes"
    DESCRIPTION = "Uses Cerebras API to generate text from language models with conversation context."

    def process_completion_request(self, model, preset, system_message, user_input, temperature, max_tokens, top_p, seed, stop, json_mode, conversation_id):

        # --- Check if model list was fetched successfully ---
        if "error_fetching_models" in self.instance_llm_models: # Use instance list here
            return ("Error fetching model list from Cerebras API. Cannot proceed.", False, conversation_id, "{}")

        # Set the seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if preset == self.DEFAULT_PROMPT:
            system_message = system_message
        else:
            system_message = get_prompt_content(self.prompt_options, preset)

        # Get or create conversation history
        if not conversation_id:
            conversation_id = self.chat_history_manager.create_new_conversation()

        conversation_history = self.chat_history_manager.get_history(conversation_id)

        # Add system message if it's a new conversation
        if not conversation_history:
            conversation_history.append({"role": "system", "content": system_message})

        # Add user input to conversation history
        conversation_history.append({"role": "user", "content": user_input})

        prompt_messages = conversation_history # API likely expects messages in this format

        try:
            # --- Cerebras API Text Generation Request ---
            inference_url = f"{self.cerebras_api_base_url}/chat/completions" # Placeholder -  VERIFY INFERENCE API ENDPOINT
            headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'} # Content-Type for JSON

            payload = { # Placeholder payload - VERIFY PAYLOAD STRUCTURE WITH CEREBRAS API DOCS
                "model": model,
                "messages": prompt_messages, # Use the conversation history as messages
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "seed": seed,
                #"stop": [stop] if stop else None, # Stop sequences - check if API expects list or string
                #"stream": False, # Streaming - if API supports it, you can add a toggle
                #"format": "json" if json_mode else "text", # JSON mode - check API support and parameter name
                # Add other API parameters as needed (e.g., top_k, presence_penalty, etc.) - CHECK API DOCS
            }
            if stop:
                payload["stop"] = [stop] # Assuming API expects stop as a list of strings

            print(f"Sending request to Cerebras API for model '{model}' with payload: {json.dumps(payload, indent=2)}")

            response = requests.post(inference_url, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses

            api_response_json = response.json()
            print(f"Cerebras API Response: {json.dumps(api_response_json, indent=2)}")

            # --- ASSUMING API RESPONSE STRUCTURE FOR COMPLETIONS ---
            # You MUST inspect the actual JSON response to extract the generated text.
            # Example: If response is like OpenAI's chat completions:
            if 'choices' in api_response_json and api_response_json['choices']:
                generated_text = api_response_json['choices'][0]['message']['content']
            else:
                raise ValueError("Unexpected API response format: No 'choices' or empty 'choices' array.")


            success = True

        except requests.exceptions.RequestException as e:
            print(Fore.RED + f"Error during Cerebras API inference: {e}" + Style.RESET_ALL)
            generated_text = f"API request error: {e}"
            success = False
        except json.JSONDecodeError as e:
            print(Fore.RED + f"Error decoding JSON response from Cerebras API: {e}" + Style.RESET_ALL)
            generated_text = f"JSON decode error: {e}"
            success = False
        except ValueError as e:
            print(Fore.RED + f"Value Error processing API response: {e}" + Style.RESET_ALL)
            generated_text = str(e) # Error message from ValueError
            success = False
        except Exception as e: # Catch any other unexpected errors
            print(Fore.RED + f"Unexpected error during Cerebras API interaction: {e}" + Style.RESET_ALL)
            generated_text = f"Unexpected error: {e}"
            success = False


        # Add assistant's response to conversation history
        if success:
            conversation_history.append({"role": "assistant", "content": generated_text})
            self.chat_history_manager.update_history(conversation_id, conversation_history)

        chat_history = json.dumps(self.chat_history_manager.get_all_conversations(), indent=2)
        return generated_text, success, conversation_id, chat_history

    def get_chat_history(self):
        all_conversations = self.chat_history_manager.get_all_conversations()
        return json.dumps(all_conversations, indent=2)