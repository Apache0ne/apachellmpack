import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .nodes.SambaNova import SambaNovaLLMNode
from .nodes.groq_api_llm import GroqAPILLM
from .nodes.cerebras import CerebrasAPILLM

NODE_CLASS_MAPPINGS = {
    "SambaNovaLLMNode": SambaNovaLLMNode,
    "GroqAPILLM": GroqAPILLM,
    "cerebrasLLMNODE": CerebrasAPILLM
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SambaNovaLLMNode": "SambaNova LLM",
    "GroqAPILLM": "Groq LLM",
    "cerebrasLLMNODE": "Cerebras LLM"
}

_all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]