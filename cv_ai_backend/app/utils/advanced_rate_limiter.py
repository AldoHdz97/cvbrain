"""
Advanced Rate Limiter with Memory Leak Fixes
CRITICAL & HIGH PRIORITY FIXES APPLIED
"""

import asyncio
import time
import hashlib
from typing import Dict, NamedTuple, Optional, Set
from collections import defaultdict, deque
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ClientType(Enum):
    """Client type classification for adaptive rate limiting"""
    HUMAN = "human"
    BOT = "bot"
    SUSPICIOUS = "suspicious"
    VIP = "vip"

@dataclass
class RateLimitResult:
    """Rate limit check result with comprehensive information"""
    allowed: bool
    current_count: int
    limit: int
    retry_after: int
    message: str
    client_type: ClientType

class ClientProfile:
    """MEMORY LEAK FIX: Enhanced client profile with automatic cleanup"""

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.request_history: deque = deque()
        self.user_agents: Set[str] = set()
        self.endpoints_accessed: Set[str] = set()
        self.total_requests = 0
        self.error_count = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.client_type = ClientType.HUMAN
        self.violation_count = 0
        self.is_blocked = False
        self.block_until = 0

    def update_activity(self, endpoint: str, user_agent: str, success: bool):
        """Update client activity with memory optimization"""
        self.last_seen = time.time()
        self.total_requests += 1
        self.endpoints_accessed.add(endpoint)

        # MEMORY FIX: Limit user agent storage
        if len(self.user_agents) < 5:
            self.user_agents.add(user_agent[:100])  # Truncate long user agents

        if not success:
            self.error_count += 1

        self._classify_client_type()

    def _classify_client_type(self):
        """Intelligent client classification"""
        # Bot indicators
        if (len(self.user_agents) > 3 or 
            self.error_count / max(self.total_requests, 1) > 0.5 or 
            len(self.endpoints_accessed) > 8):
            self.client_type = ClientType.BOT

        # Suspicious indicators
        elif (self.violation_count > 2 or self.error_count > 15):
            self.client_type = ClientType.SUSPICIOUS

        # VIP indicators (good behavior)
        elif (self.total_requests > 50 and 
              self.error_count / self.total_requests < 0.1):
            self.client_type = ClientType.VIP

class AdvancedRateLimiter:
    """
    CRITICAL FIX: Advanced rate limiter with memory leak prevention
    - Automatic client cleanup
    - Adaptive rate limits based on client behavior
    - Memory usage monitoring and limits
    - Thread-safe operations
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 30,
        window_minutes: int = 1,
        block_duration_minutes: int = 15,
        max_clients: int = 10000  # MEMORY PROTECTION
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_seconds = window_minutes * 60
        self.block_duration = block_duration_minutes * 60
        self.max_clients = max_clients

        # MEMORY FIX: Client tracking with size limits
        self.clients: Dict[str, ClientProfile] = {}
        self._lock = asyncio.Lock()

        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_background_cleanup()

        logger.info(f"Advanced rate limiter initialized: {requests_per_minute}/min, max clients: {max_clients}")

    def _start_background_cleanup(self):
        """Start background cleanup task"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """MEMORY LEAK FIX: Periodic cleanup of inactive clients"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_inactive_clients()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

    async def _cleanup_inactive_clients(self):
        """Remove inactive clients to prevent memory explosion"""
        async with self._lock:
            current_time = time.time()
            inactive_threshold = 3600  # 1 hour
            clients_to_remove = []

            for client_id, profile in list(self.clients.items()):
                if current_time - profile.last_seen > inactive_threshold:
                    clients_to_remove.append(client_id)

            for client_id in clients_to_remove:
                del self.clients[client_id]

            # Emergency cleanup if at memory limit
            if len(self.clients) > self.max_clients:
                # Remove oldest clients
                sorted_clients = sorted(
                    self.clients.items(),
                    key=lambda x: x[1].last_seen
                )

                excess_count = len(self.clients) - self.max_clients
                for client_id, _ in sorted_clients[:excess_count]:
                    del self.clients[client_id]

            if clients_to_remove:
                logger.debug(f"Cleaned up {len(clients_to_remove)} inactive clients")

    def _get_client_profile(self, client_id: str) -> ClientProfile:
        """Get or create client profile with memory protection"""
        if client_id not in self.clients:
            # Check memory limit before creating new client
            if len(self.clients) >= self.max_clients:
                # Force cleanup of oldest client
                oldest_client = min(self.clients.values(), key=lambda c: c.last_seen)
                del self.clients[oldest_client.client_id]

            self.clients[client_id] = ClientProfile(client_id)

        return self.clients[client_id]

    def _get_adaptive_limits(self, profile: ClientProfile) -> tuple[int, int]:
        """Get adaptive rate limits based on client behavior"""
        base_limit = self.requests_per_minute
        burst_limit = self.burst_limit

        # Adjust limits based on client type
        if profile.client_type == ClientType.VIP:
            base_limit = int(base_limit * 1.5)
            burst_limit = int(burst_limit * 1.5)
        elif profile.client_type == ClientType.BOT:
            base_limit = int(base_limit * 0.3)
            burst_limit = int(burst_limit * 0.3)
        elif profile.client_type == ClientType.SUSPICIOUS:
            base_limit = int(base_limit * 0.2)
            burst_limit = int(burst_limit * 0.2)

        return base_limit, burst_limit

    async def check_rate_limit(
        self,
        client_id: str,
        endpoint: str = "default",
        user_agent: str = "unknown"
    ) -> RateLimitResult:
        """COMPREHENSIVE: Rate limit check with intelligent profiling"""
        current_time = time.time()

        async with self._lock:
            # Get client profile
            profile = self._get_client_profile(client_id)

            # Check if client is blocked
            if profile.is_blocked and current_time < profile.block_until:
                remaining_block = int(profile.block_until - current_time)
                return RateLimitResult(
                    allowed=False,
                    current_count=0,
                    limit=0,
                    retry_after=remaining_block,
                    message=f"Client blocked for {remaining_block} more seconds",
                    client_type=profile.client_type
                )
            elif profile.is_blocked and current_time >= profile.block_until:
                # Unblock client
                profile.is_blocked = False
                profile.block_until = 0
                profile.violation_count = max(0, profile.violation_count - 1)

            # Get adaptive limits
            minute_limit, burst_limit = self._get_adaptive_limits(profile)

            # Clean old requests
            while (profile.request_history and 
                   current_time - profile.request_history[0] > self.window_seconds):
                profile.request_history.popleft()

            # Check current request count
            current_count = len(profile.request_history)

            # Check burst limit (last 60 seconds)
            recent_requests = sum(
                1 for req_time in profile.request_history
                if current_time - req_time < 60
            )

            # Determine if request should be allowed
            allowed = current_count < minute_limit and recent_requests < burst_limit

            if allowed:
                # Allow request
                profile.request_history.append(current_time)
                profile.update_activity(endpoint, user_agent, True)

                return RateLimitResult(
                    allowed=True,
                    current_count=current_count + 1,
                    limit=minute_limit,
                    retry_after=0,
                    message="Request allowed",
                    client_type=profile.client_type
                )
            else:
                # Block request
                profile.violation_count += 1
                profile.update_activity(endpoint, user_agent, False)

                # Check if client should be temporarily blocked
                if profile.violation_count >= 5:
                    profile.is_blocked = True
                    profile.block_until = current_time + self.block_duration
                    logger.warning(f"Client {client_id[:8]} blocked for repeated violations")

                # Calculate retry after
                if profile.request_history:
                    oldest_request = profile.request_history[0]
                    retry_after = int(self.window_seconds - (current_time - oldest_request)) + 1
                else:
                    retry_after = 60

                limit_type = "burst" if recent_requests >= burst_limit else "rate"

                return RateLimitResult(
                    allowed=False,
                    current_count=current_count,
                    limit=minute_limit,
                    retry_after=retry_after,
                    message=f"{limit_type.title()} limit exceeded ({current_count}/{minute_limit})",
                    client_type=profile.client_type
                )

    def get_stats(self) -> Dict:
        """Get comprehensive rate limiter statistics"""
        client_types = defaultdict(int)
        blocked_clients = 0

        for profile in self.clients.values():
            client_types[profile.client_type.value] += 1
            if profile.is_blocked:
                blocked_clients += 1

        return {
            "active_clients": len(self.clients),
            "max_clients": self.max_clients,
            "blocked_clients": blocked_clients,
            "client_types": dict(client_types),
            "memory_usage": f"{len(self.clients)}/{self.max_clients}"
        }

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            self._cleanup_task.cancel()
