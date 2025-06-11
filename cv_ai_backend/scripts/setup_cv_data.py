"""
CV Data Setup Script - Missing from cvbrain7L.md
Load cv.json into ChromaDB for the Ultimate CV Service
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_settings
from services.ultimate_cv_service import get_ultimate_cv_service
from utils.chromadb_manager import UltimateChromaDBManager
from utils.connection_manager import UltimateConnectionManager

async def load_cv_data():
    """Load CV data from JSON into ChromaDB"""
    
    print("🚀 Setting up CV data for Ultimate CV Service...")
    
    # Load CV JSON
    cv_path = Path(__file__).parent.parent / "data" / "cv.json"
    
    if not cv_path.exists():
        print(f"❌ CV file not found at {cv_path}")
        print("📝 Please add your cv.json file to app/data/cv.json")
        return False
    
    with open(cv_path, 'r', encoding='utf-8') as f:
        cv_data = json.load(f)
    
    print(f"✅ Loaded CV data for {cv_data['name']}")
    
    # Initialize services
    settings = get_settings()
    chromadb_manager =