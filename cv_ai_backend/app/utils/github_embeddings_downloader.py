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
        """Check if embeddings are present - COMPREHENSIVE VERSION"""
        chroma_dir = Path(self.settings.chroma_persist_dir)
    
        if not chroma_dir.exists():
            logger.info("❌ Embeddings directory doesn't exist")
            return False
    
        # Lista completa de archivos que ChromaDB puede generar
        chromadb_patterns = [
            # Base de datos principal
            "chroma.sqlite3",
            "chroma.sqlite3-wal",
            "chroma.sqlite3-shm",
            
            # Archivos de índices HNSW
            "data_level0.bin",
            "header.bin", 
            "length.bin",
            "link_lists.bin",
            
            # Archivos de metadatos
            "index_metadata.pickle",
            "id_to_uuid.pkl",
            "uuid_to_id.pkl",
            
            # Otros archivos posibles
            "*.index",
            "*.bin",
            "*.pkl",
            "*.pickle",
            "*.sqlite*"
        ]
    
        found_files = []
        all_files = []
    
        # Buscar recursivamente todos los archivos
        for item in chroma_dir.rglob("*"):
            if item.is_file():
                all_files.append(str(item.relative_to(chroma_dir)))
            
                # Verificar contra patrones conocidos
                for pattern in chromadb_patterns:
                    if pattern.startswith("*"):
                        # Pattern con wildcard
                        if item.name.endswith(pattern[1:]) or item.suffix == pattern[1:]:
                            found_files.append(str(item.relative_to(chroma_dir)))
                            break
                    else:
                        # Pattern exacto
                        if item.name == pattern:
                            found_files.append(str(item.relative_to(chroma_dir)))
                            break
    
        # Log de todos los archivos encontrados para debug
        logger.info(f"📁 All files in embeddings directory: {all_files}")
        logger.info(f"🔍 ChromaDB files found: {found_files}")
    
        # Criterios para considerar que los embeddings están presentes
        has_sqlite = any("sqlite" in f for f in found_files)
        has_bin_files = any(".bin" in f for f in found_files)
        has_any_chromadb = len(found_files) > 0
    
        if has_sqlite or (has_bin_files and len(found_files) >= 2):
            logger.info(f"✅ ChromaDB files detected: {len(found_files)} files")
            return True
        elif has_any_chromadb:
            logger.warning(f"⚠️ Some ChromaDB files found but may be incomplete: {found_files}")
            return True  # Intentar usar lo que hay
        else:
            logger.info(f"❌ No ChromaDB files detected in {len(all_files)} total files")
            return False

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
