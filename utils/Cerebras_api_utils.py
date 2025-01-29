import json
import logging
from colorama import init, Fore, Style
import os
import requests # Import requests here as it's used in fetch_cerebras_models

init()  # Initialize colorama

logger = logging.getLogger(__name__)

def load_prompt_options(prompt_files):
    """
    Loads prompt options from JSON files.

    Args:
        prompt_files (list): A list of file paths to JSON prompt files.

    Returns:
        dict: A dictionary of prompt options, where keys are prompt names and values are prompt contents.
               Returns an empty dictionary if no prompts are loaded or in case of errors.
    """
    prompt_options = {}
    for json_file in prompt_files:
        if os.path.exists(json_file): # Check if file exists before trying to open
            try:
                with open(json_file, 'r') as file:
                    prompts = json.load(file)
                    if isinstance(prompts, list): # Handle list of prompts (original format)
                        prompt_options.update({prompt['name']: prompt['content'] for prompt in prompts})
                    elif isinstance(prompts, dict): # Handle dictionary of prompts (maybe easier to edit directly)
                        prompt_options.update(prompts)
                    else:
                        logger.warning(f"Unexpected prompt format in {json_file}. Expected list or dict.")
                logger.debug(f"Prompts loaded from {json_file}: {list(prompt_options.keys())}")
            except json.JSONDecodeError as e:
                print(Fore.RED + f"JSONDecodeError loading prompts from {json_file}: {e}" + Style.RESET_ALL)
                logger.error(f"JSONDecodeError loading prompts from {json_file}: {e}")
            except FileNotFoundError:
                print(Fore.YELLOW + f"Prompt file not found: {json_file}. Skipping." + Style.RESET_ALL)
                logger.warning(f"Prompt file not found: {json_file}. Skipping.")
            except Exception as e:
                print(Fore.RED + f"Error loading prompts from {json_file}: {e}" + Style.RESET_ALL)
                logger.error(f"Error loading prompts from {json_file}: {e}")
        else:
            print(Fore.YELLOW + f"Prompt file does not exist: {json_file}. Skipping." + Style.RESET_ALL)
            logger.warning(f"Prompt file does not exist: {json_file}. Skipping.")
    return prompt_options

def get_prompt_content(prompt_options, prompt_name):
    """
    Retrieves the content of a prompt from the prompt options dictionary.

    Args:
        prompt_options (dict): Dictionary of loaded prompt options.
        prompt_name (str): Name of the prompt to retrieve.

    Returns:
        str: Content of the prompt if found, otherwise "No content found for selected prompt".
    """
    content = prompt_options.get(prompt_name)
    if content:
        logger.debug(f"Retrieved content for prompt '{prompt_name}': {content[:50]}...") # Log first 50 chars
        return content
    else:
        error_message = f"No content found for prompt '{prompt_name}'."
        print(Fore.YELLOW + error_message + Style.RESET_ALL) # Inform user in console
        logger.warning(error_message)
        return "No content found for selected prompt" # Default return, matches original logic

def fetch_cerebras_models(api_key, cerebras_api_base_url):
    """Fetches the list of available models from the Cerebras API.

    Args:
        api_key (str): Cerebras API key for authorization.
        cerebras_api_base_url (str): Base URL of the Cerebras API.

    Returns:
        list: A list of model names fetched from the API.
              Returns an empty list if there's an error fetching models.
    """
    models_url = f"{cerebras_api_base_url}/models"
    headers = {'Authorization': f'Bearer {api_key}'} # Assuming API key auth

    try:
        response = requests.get(models_url, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        models_data = response.json()

        model_names = []
        # --- ASSUMING API RESPONSE STRUCTURE ---
        # You MUST inspect the actual JSON response from the API to get the model names.
        # Example: If models are listed in a 'data' list with 'id' field as model name:
        if 'data' in models_data and isinstance(models_data['data'], list):
            for model_item in models_data['data']:
                if 'id' in model_item:
                    model_names.append(model_item['id'])
        else:
            print(Fore.YELLOW + "Unexpected model API response format. Cannot extract model names." + Style.RESET_ALL)
            return [] # Return empty list if format is unexpected

        print(f"Fetched Cerebras models from API: {model_names}")
        return model_names

    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error fetching Cerebras models from API: {e}" + Style.RESET_ALL)
        return [] # Return empty list on error