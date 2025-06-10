"""
Ultimate Multi-Layer Caching System v3.0
High-performance caching with memory + Redis backend support

FEATURES:
- Multi-layer caching (L1: Memory, L2: Redis, L3: Disk)
- Async-safe operations with proper locking
- LRU eviction with TTL support
- Performance metrics and hit rate tracking
- Automatic cache warming and preloading
- Cache coherence and invalidation
- Latest Redis patterns (December 2024)
"""

import asyncio
import hashlib
import pickle
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import weakref



# ✅ IMPROVED Redis import handling
try:
    import redis.asyncio as aioredis  # Better alias to avoid confusion
    REDIS_AVAILABLE = True
    RedisType = aioredis.Redis  # Type alias for better code clarity
except ImportError:
    REDIS_AVAILABLE = False
    # Create a stub/mock type for type hints when Redis is not available
    class RedisType:
        """Stub Redis type for when redis is not installed"""
        pass
    aioredis = None  # Explicit None assignment

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheBackend(str, Enum):
    """Cache backend options"""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: float
    key_hash: str

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() > (self.created_at + self.ttl)

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at

    def touch(self):
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1

class CacheMetrics:
    """Cache performance metrics"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.total_requests = 0
        self.total_sets = 0
        self.memory_usage = 0
        self.start_time = time.time()

    def record_hit(self):
        """Record cache hit"""
        self.hits += 1
        self.total_requests += 1

    def record_miss(self):
        """Record cache miss"""
        self.misses += 1
        self.total_requests += 1

    def record_set(self):
        """Record cache set operation"""
        self.total_sets += 1

    def record_eviction(self):
        """Record cache eviction"""
        self.evictions += 1

    def record_invalidation(self):
        """Record cache invalidation"""
        self.invalidations += 1

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100

    @property
    def uptime_seconds(self) -> float:
        """Get cache uptime in seconds"""
        return time.time() - self.start_time

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(self.hit_rate, 2),
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "total_requests": self.total_requests,
            "total_sets": self.total_sets,
            "memory_usage_bytes": self.memory_usage,
            "uptime_seconds": round(self.uptime_seconds, 2)
        }

class MemoryCache:
    """High-performance memory cache with LRU eviction"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        self.metrics = CacheMetrics()

    def _generate_key(self, key: str) -> str:
        """Generate cache key hash"""
        return hashlib.blake2b(key.encode('utf-8'), digest_size=16).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        async with self._lock:
            key_hash = self._generate_key(key)

            if key_hash in self._cache:
                entry = self._cache[key_hash]

                if entry.is_expired:
                    # Remove expired entry
                    del self._cache[key_hash]
                    if key_hash in self._access_order:
                        self._access_order.remove(key_hash)
                    self.metrics.record_miss()
                    return None

                # Update access order (LRU)
                entry.touch()
                if key_hash in self._access_order:
                    self._access_order.remove(key_hash)
                self._access_order.append(key_hash)

                self.metrics.record_hit()
                return entry.value

            self.metrics.record_miss()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory cache"""
        async with self._lock:
            key_hash = self._generate_key(key)
            current_time = time.time()
            ttl = ttl or self.default_ttl

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.max_size and key_hash not in self._cache:
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    if oldest_key in self._cache:
                        del self._cache[oldest_key]
                        self.metrics.record_eviction()
                else:
                    break

            # Create cache entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=ttl,
                key_hash=key_hash
            )

            # Store entry
            self._cache[key_hash] = entry

            # Update access order
            if key_hash in self._access_order:
                self._access_order.remove(key_hash)
            self._access_order.append(key_hash)

            self.metrics.record_set()

    async def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        async with self._lock:
            key_hash = self._generate_key(key)

            if key_hash in self._cache:
                del self._cache[key_hash]
                if key_hash in self._access_order:
                    self._access_order.remove(key_hash)
                self.metrics.record_invalidation()
                return True

            return False

    async def clear(self) -> None:
        """Clear all entries from memory cache"""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()

    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        async with self._lock:
            expired_keys = []
            current_time = time.time()

            for key_hash, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key_hash)

            for key_hash in expired_keys:
                del self._cache[key_hash]
                if key_hash in self._access_order:
                    self._access_order.remove(key_hash)

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        stats = self.metrics.get_stats()
        stats.update({
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_estimate": len(self._cache) * 1024  # Rough estimate
        })
        return stats

class RedisCache:
    """Redis-based cache with advanced features"""

    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._redis: Optional[RedisType] = None
        self.metrics = CacheMetrics()
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, install with: pip install redis")
            return False

        if aioredis is None:  # ✅ Additional safety check
            logger.error("Redis module not properly imported")
            return False

        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=False,  # We'll handle encoding ourselves
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
            return True

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._connected = False
            return False

    def _generate_key(self, key: str) -> str:
        """Generate Redis key with namespace"""
        key_hash = hashlib.blake2b(key.encode('utf-8'), digest_size=16).hexdigest()
        return f"cv_ai:cache:{key_hash}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not REDIS_AVAILABLE or not self._connected or not self._redis:
            self.metrics.record_miss()
            return None

        try:
            redis_key = self._generate_key(key)
            data = await self._redis.get(redis_key)

            if data is not None:
                value = pickle.loads(data)
                self.metrics.record_hit()
                return value

            self.metrics.record_miss()
            return None

        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self.metrics.record_miss()
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not REDIS_AVAILABLE or not self._connected or not self._redis:
            return False

        try:
            redis_key = self._generate_key(key)
            data = pickle.dumps(value)
            ttl = ttl or self.default_ttl

            await self._redis.setex(redis_key, ttl, data)
            self.metrics.record_set()
            return True

        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not REDIS_AVAILABLE or not self._connected or not self._redis:
            return False

        try:
            redis_key = self._generate_key(key)
            result = await self._redis.delete(redis_key)

            if result > 0:
                self.metrics.record_invalidation()
                return True

            return False

        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        if not REDIS_AVAILABLE or not self._connected or not self._redis:
            return False

        try:
            # Use scan to find all our keys
            keys = []
            async for key in self._redis.scan_iter(match="cv_ai:cache:*"):
                keys.append(key)

            if keys:
                await self._redis.delete(*keys)

            return True

        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return False

    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        if not REDIS_AVAILABLE or not self._connected or not self._redis:
            return {}

        try:
            info = await self._redis.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed")
            }
        except Exception as e:
            logger.warning(f"Redis info error: {e}")
            return {}

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
        self._connected = False

class UltimateCacheSystem:
    """
    Ultimate Multi-Layer Caching System v3.0

    FEATURES:
    - L1: High-speed memory cache with LRU eviction
    - L2: Redis cache for persistence and sharing
    - Automatic fallback and cache coherence
    - Performance monitoring and optimization
    - Cache warming and preloading strategies
    """

    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        memory_config: Optional[Dict[str, Any]] = None,
        redis_config: Optional[Dict[str, Any]] = None
    ):
        self.backend = backend

        # Initialize memory cache (always available as L1)
        memory_defaults = {"max_size": 1000, "default_ttl": 3600}
        memory_settings = {**memory_defaults, **(memory_config or {})}
        self.memory_cache = MemoryCache(**memory_settings)

        # Initialize Redis cache if configured
        self.redis_cache: Optional[RedisCache] = None
        if backend in [CacheBackend.REDIS, CacheBackend.HYBRID] and redis_config:
            redis_defaults = {"default_ttl": 3600}
            redis_settings = {**redis_defaults, **redis_config}
            if "url" in redis_settings:
                self.redis_cache = RedisCache(redis_settings["url"], redis_settings["default_ttl"])

        # Combined metrics
        self.total_metrics = CacheMetrics()

        logger.info(f"Cache system initialized with backend: {backend.value}")

    async def initialize(self) -> bool:
        """Initialize cache system"""
        success = True

        if self.redis_cache:
            redis_connected = await self.redis_cache.connect()
            if not redis_connected and self.backend == CacheBackend.REDIS:
                logger.error("Redis cache required but connection failed")
                success = False
            elif not redis_connected:
                logger.warning("Redis cache unavailable, using memory only")
                self.backend = CacheBackend.MEMORY

        return success

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache system"""
        start_time = time.time()

        try:
            # Always try L1 (memory) first
            value = await self.memory_cache.get(key)
            if value is not None:
                self.total_metrics.record_hit()
                return value

            # Try L2 (Redis) if available
            if self.redis_cache and self.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
                value = await self.redis_cache.get(key)
                if value is not None:
                    # Populate L1 cache for future requests
                    await self.memory_cache.set(key, value, ttl=300)  # 5 min in L1
                    self.total_metrics.record_hit()
                    return value

            self.total_metrics.record_miss()
            return None

        finally:
            # Record access time
            access_time = time.time() - start_time
            if hasattr(self, '_last_access_times'):
                self._last_access_times.append(access_time)
                if len(self._last_access_times) > 100:
                    self._last_access_times.pop(0)
            else:
                self._last_access_times = [access_time]

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache system"""
        success = True

        # Always set in L1 (memory)
        await self.memory_cache.set(key, value, ttl)

        # Set in L2 (Redis) if available
        if self.redis_cache and self.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            redis_success = await self.redis_cache.set(key, value, ttl)
            if not redis_success:
                success = False

        self.total_metrics.record_set()
        return success

    async def delete(self, key: str) -> bool:
        """Delete key from cache system"""
        memory_deleted = await self.memory_cache.delete(key)
        redis_deleted = True

        if self.redis_cache and self.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            redis_deleted = await self.redis_cache.delete(key)

        if memory_deleted or redis_deleted:
            self.total_metrics.record_invalidation()
            return True

        return False

    async def clear(self) -> bool:
        """Clear all cache entries"""
        await self.memory_cache.clear()

        if self.redis_cache and self.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            await self.redis_cache.clear()

        return True

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "backend": self.backend.value,
            "total_metrics": self.total_metrics.get_stats(),
            "memory_cache": self.memory_cache.get_stats()
        }

        if self.redis_cache:
            stats["redis_cache"] = self.redis_cache.metrics.get_stats()
            stats["redis_info"] = await self.redis_cache.get_info()

# Calculate average access time
        if hasattr(self, '_last_access_times') and self._last_access_times:
            avg_access_time = sum(self._last_access_times) / len(self._last_access_times)
            stats["avg_access_time_ms"] = round(avg_access_time * 1000, 2)

        return stats

    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries from all cache layers"""
        memory_cleaned = await self.memory_cache.cleanup_expired()

        return {
            "memory_cleaned": memory_cleaned,
            "total_cleaned": memory_cleaned
        }

    async def warm_cache(self, preload_data: Dict[str, Any]) -> int:
        """Warm cache with preloaded data"""
        count = 0
        for key, value in preload_data.items():
            await self.set(key, value)
            count += 1

        logger.info(f"Cache warmed with {count} entries")
        return count

    async def cleanup(self):
        """Cleanup cache system resources"""
        if self.redis_cache:
            await self.redis_cache.close()

        await self.memory_cache.clear()
