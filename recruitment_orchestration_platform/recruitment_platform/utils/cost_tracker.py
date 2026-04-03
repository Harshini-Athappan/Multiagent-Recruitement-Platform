"""
Groq Cost Tracking Utility
Calculates the USD cost of LLM calls based on tokens and model type.
Prices as of Groq Dec 2024 (On-Demand Tier).
"""
from typing import Dict, Any
from loguru import logger
from models.schemas import PipelineState

PRICES = {
    "llama-3.1-8b-instant":  {"input_1m": 0.05, "output_1m": 0.08},
    "llama3-8b-8192":        {"input_1m": 0.05, "output_1m": 0.08},
    "llama-3.1-70b-versatile": {"input_1m": 0.59, "output_1m": 0.79},
    "llama3-70b-8192":       {"input_1m": 0.59, "output_1m": 0.79},
}

import threading

def _run_update_cost(pipeline: PipelineState, response_metadata: Dict[str, Any], model_name: str):
    try:
        # LangChain UsageMetadata format
        token_usage = response_metadata.get("usage", {})
        if not token_usage:
            # Fallback for manual OpenAI-style metadata
            token_usage = response_metadata.get("token_usage", {})
            
        inp = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
        out = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
        
        # Calculate cost
        price = PRICES.get(model_name, PRICES["llama-3.1-8b-instant"])
        cost = (inp / 1_000_000 * price["input_1m"]) + (out / 1_000_000 * price["output_1m"])
        
        # Update pipeline metadata
        md = pipeline.metadata
        md["input_tokens"] += inp
        md["output_tokens"] += out
        md["total_tokens"] += (inp + out)
        md["total_cost_usd"] += cost
        md["llm_calls"] += 1
        
        logger.debug(f"[Billing] Call #{md['llm_calls']} — Tokens: +{inp+out} | Cost: +${cost:.6f} | Total: ${md['total_cost_usd']:.6f}")
        
    except Exception as e:
        logger.warning(f"[Billing] Failed to track cost: {e}")

def update_cost(pipeline: PipelineState, response_metadata: Dict[str, Any], model_name: str):
    """
    Extracts tokens from LangChain/Groq metadata and updates PipelineState.
    Runs in a background thread to reduce latency.
    """
    threading.Thread(target=_run_update_cost, args=(pipeline, response_metadata, model_name), daemon=True).start()

