import re
from typing import Tuple

def extract_thought_and_clean(content: str) -> Tuple[str, str]:
    """
    Extracts content between <thought> tags and returns (thought, cleaned_content).
    If no tags are found, (empty_string, original_content) is returned.
    """
    thought = ""
    cleaned_content = content
    
    thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
        cleaned_content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
    
    return thought, cleaned_content
