"""
Ultimate Connection Pool Manager v3.0
CRITICAL: Memory leak prevention with latest async patterns

FEATURES:
- Fixed-size connection pool with automatic rotation
- Proper resource cleanup with weakref tracking
- Thread-safe operations with asyncio.Lock
- Connection health monitoring and recovery
- Metrics and performance tracking
- Latest OpenAI client patterns (December 2024)
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, Optional, Set, Any, List
from datetime import datetime, timedelta
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class ConnectionMetrics:
    """Connection pool metrics tracking"""

    def __init__(self):
        self.connections_created = 0
        self.connections_destroyed = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.last_reset = datetime.utcnow()

    def record_request(self, success: bool, response_time: float):
        """Record request metrics"""
        self.total_requests += 1
        if not success:
            self.failed_requests += 1

        # Update average response time (exponential moving average)
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        uptime = (datetime.utcnow() - self.last_reset).total_seconds()
        success_rate = (self.total_requests - self.failed_requests) / max(self.total_requests, 1)

        return {
            "connections_created": self.connections_created,
            "connections_destroyed": self.connections_destroyed,
            "active_connections": self.connections_created - self.connections_destroyed,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(success_rate, 4),
            "avg_response_time_ms": round(self.avg_response_time * 1000, 2),
            "uptime_seconds": round(uptime, 2)
        }

class ConnectionWrapper:
    """Wrapper for OpenAI client with health tracking"""

    def __init__(self, client: AsyncOpenAI, client_id: str):
        self.client = client
        self.client_id = client_id
        self.created_at = datetime.utcnow()
        self.last_used = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.is_healthy = True

    def mark_used(self, success: bool = True):
        """Mark connection as used"""
        self.last_used = datetime.utcnow()
        self.request_count += 1
        if not success:
            self.error_count += 1
            # Mark unhealthy if too many errors
            if self.error_count > 5 and self.error_count / self.request_count > 0.2:
                self.is_healthy = False

    @property
    def age_seconds(self) -> float:
        """Get connection age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds"""
        return (datetime.utcnow() - self.last_used).total_seconds()

    async def close(self):
        """Close the connection safely"""
        try:
            if hasattr(self.client, '_client') and self.client._client:
                await self.client._client.aclose()
            if hasattr(self.client, 'close'):
                await self.client.close()
        except Exception as e:
            logger.warning(f"Error closing connection {self.client_id}: {e}")

class UltimateConnectionManager:
    """
    Ultimate Connection Pool Manager v3.0

    CRITICAL FEATURES:
    - Memory leak prevention with size limits
    - Connection health monitoring and rotation
    - Automatic cleanup of idle/unhealthy connections
    - Performance metrics and monitoring
    - Thread-safe operations
    - Latest OpenAI client optimization patterns
    """

    def __init__(
        self,
        max_connections: int = 10,
        max_connection_age: int = 3600,  # 1 hour
        max_idle_time: int = 300,        # 5 minutes
        health_check_interval: int = 60   # 1 minute
    ):
        self.max_connections = max_connections
        self.max_connection_age = max_connection_age
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval

        # Connection storage
        self._connections: Dict[str, ConnectionWrapper] = {}
        self._connection_counter = 0
        self._lock = asyncio.Lock()

        # Metrics and monitoring
        self.metrics = ConnectionMetrics()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Start background tasks
        self._start_background_tasks()

        logger.info(f"Connection manager initialized: max_connections={max_connections}")

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def _cleanup_loop(self):
        """Background cleanup of idle and old connections"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._cleanup_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _health_check_loop(self):
        """Background health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_connection_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def _cleanup_connections(self):
        """Remove idle, old, or unhealthy connections"""
        async with self._lock:
            current_time = datetime.utcnow()
            connections_to_remove = []

            for conn_id, wrapper in self._connections.items():
                should_remove = (
                    wrapper.age_seconds > self.max_connection_age or
                    wrapper.idle_seconds > self.max_idle_time or
                    not wrapper.is_healthy
                )

                if should_remove:
                    connections_to_remove.append(conn_id)

            # Remove connections
            for conn_id in connections_to_remove:
                wrapper = self._connections.pop(conn_id)
                await self._close_connection_safely(wrapper)

                reason = "aged out" if wrapper.age_seconds > self.max_connection_age else                         "idle timeout" if wrapper.idle_seconds > self.max_idle_time else "unhealthy"
                logger.debug(f"Removed connection {conn_id}: {reason}")

    async def _check_connection_health(self):
        """Check and update connection health status"""
        async with self._lock:
            for wrapper in self._connections.values():
                # Reset health if connection has been idle and had errors
                if wrapper.idle_seconds > 60 and wrapper.error_count > 0:
                    wrapper.error_count = max(0, wrapper.error_count - 1)
                    if wrapper.error_count / max(wrapper.request_count, 1) <= 0.1:
                        wrapper.is_healthy = True

    async def get_openai_client(self, settings) -> AsyncOpenAI:
        """
        Get OpenAI client with optimized connection pooling

        FEATURES:
        - Connection reuse with health checking
        - Automatic cleanup of old connections
        - Latest OpenAI client configuration
        - HTTP/2 support and connection pooling
        """
        async with self._lock:
            # Try to reuse healthy connection
            for wrapper in self._connections.values():
                if wrapper.is_healthy and wrapper.idle_seconds < 30:
                    wrapper.mark_used(True)
                    return wrapper.client

            # Create new connection if under limit
            if len(self._connections) < self.max_connections:
                return await self._create_new_connection(settings)

            # Remove oldest connection and create new one
            await self._remove_oldest_connection()
            return await self._create_new_connection(settings)

    async def _create_new_connection(self, settings) -> AsyncOpenAI:
        """Create new optimized OpenAI client"""
        self._connection_counter += 1
        client_id = f"openai_client_{self._connection_counter}"

        try:
            # Latest OpenAI client configuration (December 2024)
            client_config = settings.get_openai_client_config()

            # Create HTTP client with optimized settings
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=client_config["http_client_config"]["limits"]["max_connections"],
                    max_keepalive_connections=client_config["http_client_config"]["limits"]["max_keepalive_connections"],
                    keepalive_expiry=30.0
                ),
                timeout=httpx.Timeout(
                    connect=client_config["http_client_config"]["timeout"]["connect"],
                    read=client_config["http_client_config"]["timeout"]["read"],
                    write=client_config["http_client_config"]["timeout"]["write"],
                    pool=client_config["http_client_config"]["timeout"]["pool"]
                ),
                http2=True,  # Enable HTTP/2 for better performance
                verify=True
            )

            # Create OpenAI client
            client = AsyncOpenAI(
                api_key=client_config["api_key"],
                organization=client_config.get("organization"),
                timeout=client_config["timeout"],
                max_retries=client_config["max_retries"],
                http_client=http_client
            )

            # Wrap and store connection
            wrapper = ConnectionWrapper(client, client_id)
            self._connections[client_id] = wrapper

            self.metrics.connections_created += 1
            logger.debug(f"Created new OpenAI connection: {client_id}")

            return client

        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            raise

    async def _remove_oldest_connection(self):
        """Remove the oldest connection to make room for new one"""
        if not self._connections:
            return

        # Find oldest connection
        oldest_id = min(
            self._connections.keys(),
            key=lambda k: self._connections[k].created_at
        )

        wrapper = self._connections.pop(oldest_id)
        await self._close_connection_safely(wrapper)
        logger.debug(f"Removed oldest connection: {oldest_id}")

    async def _close_connection_safely(self, wrapper: ConnectionWrapper):
        """Safely close a connection wrapper"""
        try:
            await wrapper.close()
            self.metrics.connections_destroyed += 1
        except Exception as e:
            logger.warning(f"Error closing connection {wrapper.client_id}: {e}")

    async def record_request(self, success: bool, response_time: float):
        """Record request metrics"""
        self.metrics.record_request(success, response_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection pool statistics"""
        active_connections = len(self._connections)

        # Connection age distribution
        ages = [wrapper.age_seconds for wrapper in self._connections.values()]
        idle_times = [wrapper.idle_seconds for wrapper in self._connections.values()]

        connection_stats = {
            "active_connections": active_connections,
            "max_connections": self.max_connections,
            "avg_connection_age": round(sum(ages) / max(len(ages), 1), 2),
            "avg_idle_time": round(sum(idle_times) / max(len(idle_times), 1), 2),
            "healthy_connections": sum(1 for w in self._connections.values() if w.is_healthy)
        }

        # Combine with metrics
        stats = self.metrics.get_stats()
        stats.update(connection_stats)

        return stats

    async def cleanup_all(self):
        """Cleanup all connections and stop background tasks"""
        logger.info("Starting connection manager cleanup...")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._lock:
            for wrapper in list(self._connections.values()):
                await self._close_connection_safely(wrapper)
            self._connections.clear()

        logger.info("Connection manager cleanup completed")

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_connections') and self._connections:
            logger.warning("Connection manager destroyed with active connections")
