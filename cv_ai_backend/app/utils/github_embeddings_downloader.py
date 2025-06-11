"""
GitHub Embeddings Auto-Downloader for Railway Deployment
Downloads embeddings from GitHub Releases automatically
"""

import asyncio
import httpx
import zipfile
import tarfile
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

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
                
                # CRITICAL: Signal that ChromaDB needs to be reinitialized
                logger.info("🔄 ChromaDB needs to be reinitialized to load existing data")
                
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
        """Extract ZIP file with proper directory structure handling - ULTIMATE FIX"""
        def extract():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extraer primero para ver la estructura
                zip_ref.extractall(extract_to)
                logger.info(f"🔍 ZIP extracted to: {extract_to}")
                
                # Buscar chroma.sqlite3 en cualquier ubicación
                chroma_sqlite = None
                
                # Buscar recursivamente el archivo
                for root, dirs, files in os.walk(extract_to):
                    logger.info(f"🔍 Checking directory: {root}")
                    logger.info(f"🔍 Files in directory: {files}")
                    
                    # Procesar cada archivo para extraer solo el nombre
                    for file_path in files:
                        # EXTRACCIÓN MANUAL DEL NOMBRE DEL ARCHIVO
                        # Manejar tanto separadores \ como /
                        file_name = file_path.replace('\\', '/').split('/')[-1]
                        logger.info(f"🔍 Processing file: {file_path} -> filename: {file_name}")
                        
                        if file_name == 'chroma.sqlite3':
                            # Construir la ruta completa del archivo encontrado
                            full_path = Path(root) / file_path
                            logger.info(f"🎯 Attempting to find chroma.sqlite3 at: {full_path}")
                            
                            # Verificar que el archivo realmente existe
                            if full_path.exists():
                                chroma_sqlite = full_path
                                logger.info(f"✅ Found and verified chroma.sqlite3 at: {chroma_sqlite}")
                                break
                            else:
                                # Si no existe en la ruta construida, buscar más inteligentemente
                                logger.warning(f"⚠️ File not found at {full_path}, searching more intelligently...")
                                
                                # Buscar el archivo en todos los subdirectorios de root
                                for potential_file in Path(root).rglob("chroma.sqlite3"):
                                    if potential_file.exists():
                                        chroma_sqlite = potential_file
                                        logger.info(f"✅ Found chroma.sqlite3 via rglob at: {chroma_sqlite}")
                                        break
                    
                    if chroma_sqlite:
                        break
                
                # Si aún no se encontró, hacer una búsqueda global más agresiva
                if not chroma_sqlite:
                    logger.info("🔍 Global search for chroma.sqlite3...")
                    for potential_file in extract_to.rglob("chroma.sqlite3"):
                        if potential_file.exists():
                            chroma_sqlite = potential_file
                            logger.info(f"✅ Found chroma.sqlite3 via global search at: {chroma_sqlite}")
                            break
                
                # BÚSQUEDA ALTERNATIVA: Si todo falla, buscar por contenido del directorio
                if not chroma_sqlite:
                    logger.info("🔍 Alternative search: looking for sqlite3 files...")
                    for potential_file in extract_to.rglob("*.sqlite3"):
                        if potential_file.exists() and 'chroma' in potential_file.name:
                            chroma_sqlite = potential_file
                            logger.info(f"✅ Found chroma sqlite file via pattern search: {chroma_sqlite}")
                            break
                
                if chroma_sqlite and chroma_sqlite.exists():
                    # Determinar la ruta padre del archivo encontrado
                    source_dir = chroma_sqlite.parent
                    logger.info(f"📍 Source directory: {source_dir}")
                    logger.info(f"📍 Target directory: {extract_to}")
                    logger.info(f"📍 Are they equal? {source_dir == extract_to}")
                    
                    if source_dir != extract_to:
                        # Los archivos están en un subdirectorio, moverlos
                        logger.info(f"📂 MOVING ChromaDB files from {source_dir} to {extract_to}")
                        
                        # Mover chroma.sqlite3
                        dest_sqlite = extract_to / 'chroma.sqlite3'
                        if dest_sqlite.exists():
                            dest_sqlite.unlink()
                            logger.info(f"🗑️ Removed existing chroma.sqlite3")
                        
                        shutil.move(str(chroma_sqlite), str(dest_sqlite))
                        logger.info(f"✅ Moved chroma.sqlite3 to root directory")
                        
                        # Buscar y mover directorios UUID - búsqueda mejorada
                        uuid_dirs = []
                        for item in extract_to.rglob("*"):
                            if (item.is_dir() and 
                                len(item.name) == 36 and 
                                item.name.count('-') == 4 and
                                item != extract_to):  # UUID format y no es el directorio raíz
                                uuid_dirs.append(item)
                        
                        logger.info(f"📂 Found UUID directories: {[str(d) for d in uuid_dirs]}")
                        
                        for uuid_dir in uuid_dirs:
                            dest = extract_to / uuid_dir.name
                            if dest.exists() and dest != uuid_dir:
                                shutil.rmtree(dest)
                                logger.info(f"🗑️ Removed existing directory: {dest}")
                            
                            if uuid_dir != dest:
                                shutil.move(str(uuid_dir), str(dest))
                                logger.info(f"✅ Moved UUID directory: {uuid_dir.name}")
                        
                        # Limpiar directorios temporales vacíos - versión mejorada
                        for temp_name in ['embeddings', 'chroma']:
                            temp_path = extract_to / temp_name
                            if temp_path.exists() and temp_path.is_dir():
                                try:
                                    # Verificar si está vacío recursivamente
                                    if not any(temp_path.rglob("*")):
                                        shutil.rmtree(temp_path)
                                        logger.info(f"✅ Cleaned up empty directory: {temp_name}")
                                    else:
                                        # Si no está vacío, intentar mover su contenido
                                        logger.info(f"🔄 Moving content from {temp_name} directory...")
                                        for item in temp_path.rglob("*"):
                                            if item.is_file():
                                                relative_path = item.relative_to(temp_path)
                                                dest_file = extract_to / relative_path.name
                                                if not dest_file.exists():
                                                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                                                    shutil.move(str(item), str(dest_file))
                                                    logger.info(f"✅ Moved file: {relative_path.name}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Could not clean up {temp_name}: {e}")
                    else:
                        logger.info("✅ Files already in correct location, no moving needed")
                else:
                    logger.warning("❌ No chroma.sqlite3 file found in extracted content")
                    # Log para debug: mostrar todos los archivos encontrados
                    all_files = list(extract_to.rglob("*"))
                    logger.info(f"🔍 All files found in extraction: {[str(f) for f in all_files if f.is_file()]}")
                
                # Log estado final
                final_files = []
                for item in extract_to.iterdir():
                    if item.is_file():
                        final_files.append(item.name)
                    elif item.is_dir():
                        final_files.append(f"{item.name}/")
                
                logger.info(f"📁 Final items in extract directory: {final_files}")
        
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
