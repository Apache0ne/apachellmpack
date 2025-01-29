import os
import sys

# Add the SambaNova directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .nodes.SambaNova import SambaNovaLLMNode
from .nodes.groq_api_llm import GroqAPILLM

NODE_CLASS_MAPPINGS = {
    "SambaNovaLLMNode": SambaNovaLLMNode,
    "GroqAPILLM": GroqAPILLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SambaNovaLLMNode": "SambaNova LLM",
    "GroqAPILLM": "Groq LLM"
}

_all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]