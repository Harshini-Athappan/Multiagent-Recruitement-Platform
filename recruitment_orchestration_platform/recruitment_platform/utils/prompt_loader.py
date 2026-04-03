import json
import os
from functools import lru_cache
from typing import Dict, Any

@lru_cache()
def load_agent_prompt(agent_name: str) -> Dict[str, Any]:
    """Loads a specific agent's JSON prompt file."""
    # Updated path to look in agents/prompts/
    path = os.path.join(os.path.dirname(__file__), f"../agents/prompts/{agent_name}.json")
    if not os.path.exists(path):
        # Fallback to legacy path if not found
        legacy_path = os.path.join(os.path.dirname(__file__), "../prompts/agent_prompts.json")
        if os.path.exists(legacy_path):
            with open(legacy_path, "r") as f:
                all_prompts = json.load(f)
                return all_prompts.get(agent_name, {})
        raise FileNotFoundError(f"Prompt file not found for agent: {agent_name}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_system_prompt(agent_name: str) -> str:
    """Returns the 'system_prompt' field for a given agent."""
    prompt_data = load_agent_prompt(agent_name)
    return prompt_data.get("system_prompt", "")


def get_agent_prompt_template(agent_name: str, key: str) -> str:
    """Returns a specific prompt template string from an agent's JSON."""
    prompt_data = load_agent_prompt(agent_name)
    return prompt_data.get(key, "")
