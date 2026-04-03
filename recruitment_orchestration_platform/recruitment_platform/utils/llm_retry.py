"""
Shared LLM retry utility.
Wraps LangChain LLM invoke() with exponential backoff for Groq rate limits.
"""
import time
from loguru import logger


def llm_invoke_with_retry(llm, messages, max_retries: int = 4, base_wait: float = 5.0):
    """
    Invoke the LLM with automatic retry on rate-limit (429) errors.
    Uses exponential backoff: 5s, 10s, 20s, 40s.

    Args:
        llm: LangChain LLM instance (e.g. ChatGroq)
        messages: list of LangChain messages
        max_retries: maximum number of retry attempts
        base_wait: base wait time in seconds for backoff

    Returns:
        LLM response object

    Raises:
        Exception: if all retries are exhausted
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_rate_limit = "rate_limit" in err_str or "429" in err_str or "too many" in err_str or "rate limit" in err_str
            if is_rate_limit and attempt < max_retries:
                wait = base_wait * (2 ** attempt)
                logger.warning(f"[LLM] Rate limit hit (attempt {attempt+1}/{max_retries}). Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise
    raise last_err
