"""
GitHub Embeddings Auto-Downloader for Railway Deployment
Downloads embeddings from GitHub Releases automatically
"""

import asyncio
import httpx
import zipfile
import tarfile
import logging
from pathlib import Path
from typing import Optional
import os

logger = logging.getLogger(__name__)

class GitHubEmbeddingsDownloader:
    """
    Downloads CV embeddings from GitHub Releases automatically
    
    Perfect for Railway deployment where large files can't be in repo
    """
    
    def __init__(self, settings):
        self.settings = settings
        ##########################Cambiar estas variables por variables de entorno en Railway- Es una opcion
        # Configuration
        self.github_user = "AldoHdz97"
        self.github_repo = "cvbrain"
        self.release_version = "v1.0.0"
        self.filename = "cv_embeddings.zip"
        
        # Build download URL
        self.download_url = (
            f"https://github.com/{self.github_user}/{self.github_repo}/"
            f"releases/download/{self.release_version}/{self.filename}"
        )
        
        logger.info(f"GitHub Embeddings Downloader initialized")
        logger.info(f"Download URL: {self.download_url}")
    
    def _check_embeddings_exist(self) -> bool:
        """Check if embeddings are already present"""
        chroma_dir = Path(self.settings.chroma_persist_dir)
        
        # Check for key ChromaDB files
        required_files = [
            "chroma.sqlite3",
            # Could also check for specific collection folders
        ]
        
        for file in required_files:
            if not (chroma_dir / file).exists():
                logger.info(f"Missing embedding file: {file}")
                return False
        
        logger.info("✅ Embeddings already exist")
        return True
    
    async def download_and_extract(self) -> bool:
        """
        Download embeddings from GitHub Releases and extract them
        
        Returns:
            bool: True if successful, False otherwise
        """
        
        # Check if already exists
        if self._check_embeddings_exist():
            return True
        
        try:
            logger.info("📥 Downloading embeddings from GitHub Releases...")
            logger.info(f"URL: {self.download_url}")
            
            # Create directory
            embed_dir = Path(self.settings.chroma_persist_dir)
            embed_dir.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),  # 5 minutes timeout
                follow_redirects=True
            ) as client:
                
                response = await client.get(self.download_url)
                response.raise_for_status()
                
                file_size = len(response.content)
                logger.info(f"Downloaded {file_size / (1024*1024):.1f} MB")
            
            # Save to temporary file
            temp_file = embed_dir / f"temp_{self.filename}"
            with open(temp_file, "wb") as f:
                f.write(response.content)
            
            logger.info("📂 Extracting embeddings...")
            
            # Extract based on file type
            if self.filename.endswith('.zip'):
                await self._extract_zip(temp_file, embed_dir)
            elif self.filename.endswith('.tar.gz'):
                await self._extract_tar(temp_file, embed_dir)
            else:
                raise ValueError(f"Unsupported file format: {self.filename}")
            
            # Clean up temp file
            temp_file.unlink()
            
            # Verify extraction
            if self._check_embeddings_exist():
                logger.info("✅ Embeddings downloaded and extracted successfully!")
                return True
            else:
                logger.error("❌ Extraction verification failed")
                return False
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.error(f"❌ Release not found: {self.download_url}")
                logger.error("Make sure you've created the GitHub release and uploaded the file")
            else:
                logger.error(f"❌ HTTP error downloading embeddings: {e}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to download embeddings: {e}")
            return False
    
    async def _extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract ZIP file"""
        def extract():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        
        # Run in thread to avoid blocking
        await asyncio.to_thread(extract)
    
    async def _extract_tar(self, tar_path: Path, extract_to: Path):
        """Extract TAR.GZ file"""
        def extract():
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_to)
        
        # Run in thread to avoid blocking
        await asyncio.to_thread(extract)
    
    def get_status(self) -> dict:
        """Get current status of embeddings"""
        return {
            "embeddings_exist": self._check_embeddings_exist(),
            "download_url": self.download_url,
            "target_directory": str(self.settings.chroma_persist_dir)
        }

# Helper function for easy integration
async def ensure_embeddings_available(settings) -> bool:
    """
    Ensure embeddings are available, download if necessary
    
    Args:
        settings: Application settings
        
    Returns:
        bool: True if embeddings are available, False otherwise
    """
    downloader = GitHubEmbeddingsDownloader(settings)
    return await downloader.download_and_extract()
