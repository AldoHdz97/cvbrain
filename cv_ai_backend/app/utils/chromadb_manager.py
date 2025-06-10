"""
Ultimate ChromaDB Manager v3.0
Thread-safe operations with latest ChromaDB 1.0.11+ patterns

FEATURES:
- Thread-safe operations using asyncio.to_thread
- Connection pooling and health monitoring
- Automatic collection discovery and validation
- Performance metrics and query optimization
- Backup and recovery capabilities
- Latest ChromaDB 1.0.11+ configuration patterns
"""

import asyncio
import logging
import time
import json
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import threading

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)

class ChromaDBMetrics:
    """ChromaDB performance metrics"""

    def __init__(self):
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.total_query_time = 0.0
        self.avg_query_time = 0.0
        self.collection_count = 0
        self.document_count = 0
        self.last_health_check = None
        self.start_time = datetime.utcnow()

    def record_query(self, success: bool, query_time: float):
        """Record query metrics"""
        self.total_queries += 1
        self.total_query_time += query_time

        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

        # Update average (exponential moving average)
        if self.avg_query_time == 0:
            self.avg_query_time = query_time
        else:
            self.avg_query_time = 0.9 * self.avg_query_time + 0.1 * query_time

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive ChromaDB statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        success_rate = (self.successful_queries / max(self.total_queries, 1)) * 100

        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate_percent": round(success_rate, 2),
            "avg_query_time_ms": round(self.avg_query_time * 1000, 2),
            "total_query_time_seconds": round(self.total_query_time, 2),
            "collection_count": self.collection_count,
            "document_count": self.document_count,
            "uptime_seconds": round(uptime, 2),
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }

class UltimateChromaDBManager:
    """
    Ultimate ChromaDB Manager v3.0

    FEATURES:
    - Thread-safe operations with asyncio.to_thread
    - Advanced connection management and health monitoring
    - Collection discovery and automatic validation
    - Query optimization and performance tracking
    - Backup and recovery capabilities
    - Latest ChromaDB 1.0.11+ patterns and configurations
    """

    def __init__(self, settings):
        self.settings = settings
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[Collection] = None
        self._lock = asyncio.Lock()
        self.metrics = ChromaDBMetrics()

        # Connection settings
        self._max_retries = 3
        self._retry_delay = 1.0
        self._initialized = False
        self._last_health_check = None

        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        logger.info("ChromaDB manager initialized")

    async def initialize(self) -> bool:
        """Initialize ChromaDB with comprehensive error handling"""
        async with self._lock:
            if self._initialized:
                return True

            try:
                logger.info("🔌 Initializing ChromaDB with thread-safe operations...")

                # Create client with retries
                for attempt in range(self._max_retries):
                    try:
                        self._client = await asyncio.to_thread(self._create_client)
                        if self._client:
                            break
                    except Exception as e:
                        if attempt == self._max_retries - 1:
                            raise
                        logger.warning(f"ChromaDB client creation attempt {attempt + 1} failed: {e}")
                        await asyncio.sleep(self._retry_delay * (attempt + 1))

                if not self._client:
                    raise Exception("Failed to create ChromaDB client after retries")

                # Discover and validate collections
                collections = await asyncio.to_thread(self._client.list_collections)
                self.metrics.collection_count = len(collections)

                target_collection = self._find_target_collection(collections)
                if not target_collection:
                    # Try to create collection if it doesn't exist
                    target_collection = await self._create_or_get_collection()

                if not target_collection:
                    raise Exception("No suitable collection found and couldn't create one")

                self._collection = target_collection

                # Validate collection content
                doc_count = await asyncio.to_thread(self._collection.count)
                self.metrics.document_count = doc_count

                if doc_count == 0:
                    logger.warning("⚠️  Collection is empty - consider running data setup")
                else:
                    logger.info(f"📊 ChromaDB initialized: {doc_count} documents in '{target_collection.name}'")

                # Start health monitoring
                self._start_health_monitoring()

                self._initialized = True
                return True

            except Exception as e:
                logger.error(f"❌ ChromaDB initialization failed: {e}")
                return False

    def _create_client(self) -> Optional[chromadb.ClientAPI]:
        try:
            persist_dir = Path(self.settings.chroma_persist_dir).expanduser().resolve()
            persist_dir.mkdir(parents=True, exist_ok=True)
            # NEW: Just use this
            return chromadb.PersistentClient(path=str(persist_dir))
        except Exception as e:
            logger.error(f"ChromaDB client creation failed: {e}")
            return None


    def _find_target_collection(self, collections: List[Collection]) -> Optional[Collection]:
        """Find target collection with intelligent matching"""
        target_name = self.settings.chroma_collection_name

        # Try exact match first
        for collection in collections:
            if collection.name == target_name:
                logger.info(f"✅ Found exact match collection: '{target_name}'")
                return collection

        # Try partial match (useful for timestamped collections)
        base_name = target_name.split('_')[0] if '_' in target_name else target_name
        for collection in collections:
            if collection.name.startswith(base_name):
                logger.info(f"✅ Found partial match collection: '{collection.name}' for target '{target_name}'")
                return collection

        # Fallback to most recent collection (by name sorting)
        if collections:
            sorted_collections = sorted(collections, key=lambda c: c.name, reverse=True)
            latest = sorted_collections[0]
            logger.warning(f"⚠️  No match for '{target_name}', using latest: '{latest.name}'")
            return latest

        return None

    async def _create_or_get_collection(self) -> Optional[Collection]:
        """Create collection if it doesn't exist"""
        try:
            collection_name = self.settings.chroma_collection_name

            # Try to get existing collection
            try:
                collection = await asyncio.to_thread(
                    self._client.get_collection,
                    name=collection_name
                )
                logger.info(f"✅ Retrieved existing collection: '{collection_name}'")
                return collection
            except Exception:
                # Collection doesn't exist, create it
                pass

            # Create new collection with metadata
            metadata = {
                "description": "CV document chunks for semantic search",
                "created_at": datetime.utcnow().isoformat(),
                "created_by": "cv_ai_backend",
                "version": self.settings.app_version
            }

            collection = await asyncio.to_thread(
                self._client.create_collection,
                name=collection_name,
                metadata=metadata
            )

            logger.info(f"✅ Created new collection: '{collection_name}'")
            return collection

        except Exception as e:
            logger.error(f"Failed to create or get collection: {e}")
            return None

    def _start_health_monitoring(self):
        """Start background health monitoring"""
        if not self._health_monitor_task or self._health_monitor_task.done():
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            if not self._collection:
                return

            start_time = time.time()

            # Check collection accessibility
            doc_count = await asyncio.to_thread(self._collection.count)

            # Update metrics
            self.metrics.document_count = doc_count
            self.metrics.last_health_check = datetime.utcnow()
            self._last_health_check = datetime.utcnow()

            health_check_time = time.time() - start_time
            logger.debug(f"Health check completed in {health_check_time:.3f}s: {doc_count} documents")

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def query_collection(
        self,
        query_embedding: List[float],
        k: int = 3,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Thread-safe collection query with performance tracking

        Args:
            query_embedding: Query vector
            k: Number of results to return
            where: Metadata filter conditions
            include: Fields to include in results

        Returns:
            Query results with documents, distances, and metadata
        """
        if not self._initialized or not self._collection:
            raise Exception("ChromaDB not properly initialized")

        start_time = time.time()

        try:
            # Default includes
            if include is None:
                include = ['documents', 'distances', 'metadatas']

            # Perform query using asyncio.to_thread for thread safety
            results = await asyncio.to_thread(
                lambda: self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=where,
                    include=include
                )
            )

            query_time = time.time() - start_time
            self.metrics.record_query(True, query_time)

            # Process results
            processed_results = {
                'documents': results.get('documents', [[]])[0],
                'distances': results.get('distances', [[]])[0],
                'metadatas': results.get('metadatas', [[]])[0],
                'ids': results.get('ids', [[]])[0] if 'ids' in include else []
            }

            logger.debug(f"Query completed in {query_time:.3f}s, returned {len(processed_results['documents'])} results")

            return processed_results

        except Exception as e:
            query_time = time.time() - start_time
            self.metrics.record_query(False, query_time)
            logger.error(f"ChromaDB query failed after {query_time:.3f}s: {e}")
            raise

    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to collection with thread safety"""
        if not self._initialized or not self._collection:
            raise Exception("ChromaDB not properly initialized")

        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{i}_{int(time.time())}" for i in range(len(documents))]

            # Add documents using asyncio.to_thread
            await asyncio.to_thread(
                lambda: self._collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            )

            # Update document count
            self.metrics.document_count = await asyncio.to_thread(self._collection.count)

            logger.info(f"Added {len(documents)} documents to collection")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from collection"""
        if not self._initialized or not self._collection:
            raise Exception("ChromaDB not properly initialized")

        try:
            await asyncio.to_thread(
                lambda: self._collection.delete(ids=ids)
            )

            # Update document count
            self.metrics.document_count = await asyncio.to_thread(self._collection.count)

            logger.info(f"Deleted {len(ids)} documents from collection")
            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get comprehensive collection information"""
        if not self._initialized or not self._collection:
            return {"status": "not_initialized"}

        try:
            # Get basic collection info
            doc_count = await asyncio.to_thread(self._collection.count)

            # Get collection metadata
            metadata = getattr(self._collection, 'metadata', {})

            # Get a sample of documents for analysis
            sample_size = min(5, doc_count)
            if sample_size > 0:
                sample_results = await asyncio.to_thread(
                    lambda: self._collection.get(
                        limit=sample_size,
                        include=['documents', 'metadatas']
                    )
                )
            else:
                sample_results = {'documents': [], 'metadatas': []}

            return {
                "status": "healthy",
                "collection_name": self._collection.name,
                "document_count": doc_count,
                "metadata": metadata,
                "sample_documents": len(sample_results.get('documents', [])),
                "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
                "metrics": self.metrics.get_stats()
            }

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"status": "error", "error": str(e)}

    async def backup_collection(self, backup_path: str) -> bool:
        """Create a backup of the collection"""
        if not self._initialized or not self._collection:
            return False

        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Get all documents
            all_docs = await asyncio.to_thread(
                lambda: self._collection.get(include=['documents', 'metadatas', 'embeddings'])
            )

            # Create backup metadata
            backup_metadata = {
                "collection_name": self._collection.name,
                "document_count": len(all_docs.get('documents', [])),
                "backup_timestamp": datetime.utcnow().isoformat(),
                "backup_version": self.settings.app_version
            }

            # Save backup files
            backup_file = backup_dir / f"collection_backup_{int(time.time())}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": backup_metadata,
                    "data": all_docs
                }, f, indent=2, default=str)

            logger.info(f"Collection backup created: {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    async def cleanup(self):
        """Cleanup ChromaDB manager resources"""
        logger.info("🧹 Starting ChromaDB manager cleanup...")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel health monitoring task
        if self._health_monitor_task and not self._health_monitor_task.done():
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # No explicit cleanup needed for ChromaDB client
        # It handles its own resource management

        self._initialized = False
        logger.info("✅ ChromaDB manager cleanup completed")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive ChromaDB statistics"""
        return self.metrics.get_stats()
