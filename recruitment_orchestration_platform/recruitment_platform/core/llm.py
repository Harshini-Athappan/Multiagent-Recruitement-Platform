from langchain_groq import ChatGroq
from core.config import settings
from dotenv import load_dotenv
load_dotenv()


def get_llm(temperature: float = 0.2, max_tokens: int = 4096) -> ChatGroq:
    """
    Return a configured ChatGroq instance.
    Uses the model specified in settings (default: llama-3.3-70b-versatile).
    """
    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
