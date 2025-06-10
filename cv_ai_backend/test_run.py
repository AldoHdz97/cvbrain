#!/usr/bin/env python3
"""Simple test runner to debug startup issues"""

import os
import sys
import logging

# Add current directory to Python path
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports step by step"""
    logger.info("🧪 Testing imports...")
    
    try:
        logger.info("Testing FastAPI...")
        import fastapi
        logger.info(f"✅ FastAPI {fastapi.__version__}")
        
        logger.info("Testing Pydantic...")
        import pydantic
        logger.info(f"✅ Pydantic {pydantic.__version__}")
        
        logger.info("Testing OpenAI...")
        import openai
        logger.info(f"✅ OpenAI {openai.__version__}")
        
        logger.info("Testing ChromaDB...")
        import chromadb
        logger.info(f"✅ ChromaDB {chromadb.__version__}")
        
        logger.info("Testing app configuration...")
        from app.core.config import get_settings
        settings = get_settings()
        logger.info(f"✅ Configuration loaded: {settings.app_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_chromadb():
    """Test ChromaDB connection"""
    logger.info("🔍 Testing ChromaDB connection...")
    
    try:
        import chromadb
        from app.core.config import get_settings
        
        settings = get_settings()
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collections = client.list_collections()
        
        logger.info(f"✅ ChromaDB connected: {len(collections)} collections found")
        for collection in collections:
            count = collection.count()
            logger.info(f"   📋 {collection.name}: {count} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ChromaDB test failed: {e}")
        return False

def main():
    logger.info("🚀 CV-AI Backend Diagnostic Test")
    logger.info("=" * 40)
    
    if not test_imports():
        logger.error("❌ Import test failed")
        return
    
    if not test_chromadb():
        logger.error("❌ ChromaDB test failed")
        return
    
    logger.info("✅ All tests passed! Try running the main server now.")

if __name__ == "__main__":
    main()