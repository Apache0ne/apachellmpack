import configparser
import os
from groq import Groq
from colorama import init, Fore, Style
import requests
import json

init() 

NODE_FOLDER_PATH = os.path.dirname(os.path.dirname(__file__)) # Go up two levels from your current file
GROQ_CONFIG_PATH = os.path.join(NODE_FOLDER_PATH, 'nodes', 'groq', 'GroqConfig.ini')

def load_config(filepath):
    config = configparser.ConfigParser()
    if not os.path.exists(filepath):
        print(Fore.YELLOW + f"Configuration file {filepath} does not exist." + Style.RESET_ALL)
        return None
    config.read(filepath)
    try:
        return config['API']['key']
    except KeyError:
        print(Fore.RED + f"Key 'key' not found in {filepath} or invalid key in {filepath}." + Style.RESET_ALL)
        return None

def fetch_groq_models():
    api_key = load_config(GROQ_CONFIG_PATH)
    if not api_key:
        print(Fore.RED + "Groq API key missing or invalid, cannot fetch models." + Style.RESET_ALL)
        return []

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        models_data = response.json()

        model_ids = []
        if isinstance(models_data, dict) and 'data' in models_data and isinstance(models_data['data'], list):
            for model in models_data['data']:
                if 'id' in model:
                    model_ids.append(model['id'])
        else:
            print(Fore.YELLOW + "Unexpected Groq model API response format. Cannot extract model names." + Style.RESET_ALL)
            return []

        #print(Fore.GREEN + f"{model_ids}" + Style.RESET_ALL)
        return model_ids

    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"Error fetching Groq models from API. Check your API key and connection. Error: {e}" + Style.RESET_ALL)
        return []
    except json.JSONDecodeError as e:
        print(Fore.RED + f"Error decoding JSON response from Groq API: {e}. Check API response format. Error: {e}" + Style.RESET_ALL)
        return []