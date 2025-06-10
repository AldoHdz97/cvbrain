#!/usr/bin/env python3
"""CV-AI Backend Server Runner"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_environment():
    """Validate environment and dependencies"""
    logger.info("🔍 Checking environment...")

    # Check Python version
    if sys.version_info < (3, 9):
        logger.error(f"❌ Python 3.9+ required, found {sys.version}")
        return False

    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check required packages
    required_packages = ["fastapi", "uvicorn", "pydantic", "openai", "chromadb"]

    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}: OK")
        except ImportError:
            logger.error(f"❌ {package}: Missing")
            return False

    # Check configuration
    try:
        settings = get_settings()

        if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
            logger.error("❌ OpenAI API key not configured")
            logger.error("   Set CV_AI_OPENAI_API_KEY environment variable")
            return False

        logger.info("✅ Configuration: OK")
        return True

    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False


def main():
    """Main server runner function"""

    logger.info("🚀 CV-AI Backend Server Runner")
    logger.info("=" * 50)

    if not check_environment():
        logger.error("❌ Environment checks failed")
        logger.info("💡 Install dependencies: pip install -r requirements.txt")
        logger.info("💡 Set OpenAI key: CV_AI_OPENAI_API_KEY=your-key")
        sys.exit(1)

    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    logger.info("⚙️  Configuration:")
    logger.info(f"   Host: {settings.api_host}")
    logger.info(f"   Port: {settings.api_port}")
    logger.info(f"   Reload: {settings.api_reload}")
    logger.info(f"   OpenAI Model: {settings.openai_model}")

    logger.info("🌐 Server URLs:")
    logger.info(f"   API: http://{settings.api_host}:{settings.api_port}")
    logger.info(f"   Docs: http://{settings.api_host}:{settings.api_port}/docs")
    logger.info(f"   Health: http://{settings.api_host}:{settings.api_port}/api/v1/health")

    logger.info("=" * 50)
    logger.info("🎯 Starting CV-AI Backend Server...")
    logger.info("=" * 50)

    try:
        from app.main import run_server
        run_server()

    except KeyboardInterrupt:
        logger.info("\n👋 Server stopped by user")

    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
