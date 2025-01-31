import json
import logging
from colorama import init, Fore, Style
import os
import requests 

init()  

logger = logging.getLogger(__name__)

def load_prompt_options(prompt_files):
    prompt_options = {}
    for json_file in prompt_files:
        if os.path.exists(json_file): 
            try:
                with open(json_file, 'r') as file:
                    prompts = json.load(file)
                    if isinstance(prompts, list): 
                        prompt_options.update({prompt['name']: prompt['content'] for prompt in prompts})
                    elif isinstance(prompts, dict): 
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
    content = prompt_options.get(prompt_name)
    if content:
        logger.debug(f"Retrieved content for prompt '{prompt_name}': {content[:50]}...") 
    else:
        error_message = f"No content found for prompt '{prompt_name}'."
        print(Fore.YELLOW + error_message + Style.RESET_ALL) 
        logger.warning(error_message)
        return "No content found for selected prompt" 

def fetch_cerebras_models(api_key, cerebras_api_base_url):
    models_url = f"{cerebras_api_base_url}/models"
    headers = {'Authorization': f'Bearer {api_key}'} 

    try:
        response = requests.get(models_url, headers=headers)
        response.raise_for_status()
        models_data = response.json()

        model_names = []
        if 'data' in models_data and isinstance(models_data['data'], list):
            for model_item in models_data['data']:
                if 'id' in model_item:
                    model_names.append(model_item['id'])
        else:
            print(Fore.YELLOW + "Unexpected model API response format. Cannot extract model names." + Style.RESET_ALL)
            return []

        #print(f"Fetched Cerebras models from API: {model_names}")
        return model_names

    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error fetching Cerebras models from API: {e}" + Style.RESET_ALL)
        return [] 