"""
Entry point for the Recruitment Orchestration Platform.
Run with: python run.py
"""
import uvicorn
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import settings

from loguru import logger

# Add this to force write logs to a file so you can always see them
logger.add("app.log", rotation="10 MB", enqueue=True, level="INFO")

logger.info("🚀 Recruitment Orchestration Platform is starting up...")

if __name__ == "__main__":
    logger.info("Running Uvicorn server...")
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # <--- MUST BE FALSE ON WINDOWS to see terminal logs properly!
        log_level=settings.log_level.lower(),
    )
