"""
Ultimate Advanced Rate Limiter v3.0
Intelligent client profiling with memory leak prevention

FEATURES:
- Advanced client behavior analysis and classification
- Adaptive rate limits based on client reputation
- Memory leak prevention with automatic cleanup
- Bot detection and blocking capabilities
- VIP client recognition and preferential treatment
- Comprehensive metrics and monitoring
- Sliding window rate limiting with burst protection
"""

import asyncio
import time
import hashlib
import re
from typing import Dict, Optional, Set, List, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import weakref

logger = logging.getLogger(__name__)

class ClientType(str, Enum):
    """Client classification types"""
    UNKNOWN = "unknown"
    HUMAN = "human"
    BOT_FRIENDLY = "bot_friendly"
    BOT_AGGRESSIVE = "bot_aggressive"
    SUSPICIOUS = "suspicious"
    VIP = "vip"
    BLOCKED = "blocked"

class ViolationType(str, Enum):
    """Types of rate limit violations"""
    BURST_LIMIT = "burst_limit"
    RATE_LIMIT = "rate_limit"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    BOT_BEHAVIOR = "bot_behavior"
    SECURITY_VIOLATION = "security_violation"

@dataclass
class RateLimitResult:
    """Comprehensive rate limit check result"""
    allowed: bool
    current_count: int
    limit: int
    retry_after: int
    message: str
    client_type: ClientType
    violation_type: Optional[ViolationType] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClientViolation:
    """Record of a client violation"""
    violation_type: ViolationType
    timestamp: float
    details: Dict[str, Any]
    severity: int  # 1-10 scale

class ClientProfile:
    """
    Comprehensive client profile with behavior analysis

    FEATURES:
    - Request pattern analysis
    - User agent and endpoint tracking
    - Violation history and scoring
    - Automatic classification and reputation
    """

    def __init__(self, client_id: str):
        self.client_id = client_id
        self.created_at = time.time()
        self.last_seen = time.time()

        # Request tracking
        self.request_history: deque = deque(maxlen=1000)  # Memory protection
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Behavior analysis
        self.user_agents: Set[str] = set()
        self.endpoints_accessed: Set[str] = set()
        self.request_intervals: List[float] = []
        self.countries: Set[str] = set()

        # Classification and reputation
        self.client_type = ClientType.UNKNOWN
        self.reputation_score = 50  # 0-100 scale
        self.trust_level = 1  # 1-5 scale

        # Violations and blocking
        self.violations: List[ClientViolation] = []
        self.is_blocked = False
        self.block_until = 0
        self.block_reason = ""

        # Performance metrics
        self.avg_response_time = 0.0
        self.error_rate = 0.0

    def update_activity(
        self,
        endpoint: str,
        user_agent: str,
        success: bool,
        response_time: float = 0.0,
        country: Optional[str] = None
    ):
        """Update client activity and analyze behavior"""
        current_time = time.time()

        # Update basic metrics
        self.last_seen = current_time
        self.total_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        # Track request intervals (for bot detection)
        if self.request_history:
            interval = current_time - self.request_history[-1]
            self.request_intervals.append(interval)
            if len(self.request_intervals) > 100:  # Memory protection
                self.request_intervals.pop(0)

        # Update collections with size limits (memory protection)
        if len(self.user_agents) < 10:
            self.user_agents.add(user_agent[:200])  # Truncate long UAs

        if len(self.endpoints_accessed) < 50:
            self.endpoints_accessed.add(endpoint)

        if country and len(self.countries) < 10:
            self.countries.add(country)

        # Update request history
        self.request_history.append(current_time)

        # Update performance metrics
        if response_time > 0:
            if self.avg_response_time == 0:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = 0.9 * self.avg_response_time + 0.1 * response_time

        # Update error rate
        self.error_rate = self.failed_requests / max(self.total_requests, 1)

        # Analyze and update classification
        self._analyze_behavior()

    def _analyze_behavior(self):
        """Analyze client behavior and update classification"""
        old_type = self.client_type

        # Bot detection patterns
        bot_indicators = 0

        # Check user agent patterns
        for ua in self.user_agents:
            ua_lower = ua.lower()
            if any(pattern in ua_lower for pattern in [
                'bot', 'crawler', 'spider', 'scraper', 'python-requests',
                'curl', 'wget', 'http', 'automated'
            ]):
                bot_indicators += 2

        # Check request patterns
        if len(self.request_intervals) > 10:
            # Very regular intervals suggest bot
            intervals = sorted(self.request_intervals[-20:])
            if len(intervals) > 5:
                median_interval = intervals[len(intervals) // 2]
                regular_intervals = sum(
                    1 for interval in intervals[-10:]
                    if abs(interval - median_interval) < 0.5
                )
                if regular_intervals > 7:  # >70% regular intervals
                    bot_indicators += 3

            # Very fast requests suggest aggressive bot
            fast_requests = sum(1 for interval in intervals if interval < 0.5)
            if fast_requests > len(intervals) * 0.5:  # >50% fast requests
                bot_indicators += 2

        # Check endpoint diversity (bots often access many endpoints)
        if len(self.endpoints_accessed) > 10:
            bot_indicators += 1

        # Check error rate (high error rate suggests probing)
        if self.error_rate > 0.3:  # >30% error rate
            bot_indicators += 2

        # Multiple user agents suggest bot rotation
        if len(self.user_agents) > 3:
            bot_indicators += 1

        # Classification logic
        if bot_indicators >= 6:
            self.client_type = ClientType.BOT_AGGRESSIVE
            self.reputation_score = max(0, self.reputation_score - 10)
        elif bot_indicators >= 3:
            self.client_type = ClientType.BOT_FRIENDLY
            self.reputation_score = max(10, self.reputation_score - 5)
        elif self.error_rate > 0.5 or len(self.violations) > 3:
            self.client_type = ClientType.SUSPICIOUS
            self.reputation_score = max(0, self.reputation_score - 15)
        elif self.total_requests > 100 and self.error_rate < 0.05 and len(self.violations) == 0:
            # Good behavior for established clients
            self.client_type = ClientType.VIP
            self.reputation_score = min(100, self.reputation_score + 5)
        elif bot_indicators == 0 and self.error_rate < 0.1:
            self.client_type = ClientType.HUMAN
            self.reputation_score = min(80, self.reputation_score + 1)

        # Update trust level based on reputation
        if self.reputation_score >= 80:
            self.trust_level = 5
        elif self.reputation_score >= 60:
            self.trust_level = 4
        elif self.reputation_score >= 40:
            self.trust_level = 3
        elif self.reputation_score >= 20:
            self.trust_level = 2
        else:
            self.trust_level = 1

        # Log classification changes
        if old_type != self.client_type:
            logger.info(
                f"Client {self.client_id[:8]} reclassified: "
                f"{old_type.value} -> {self.client_type.value} "
                f"(reputation: {self.reputation_score}, trust: {self.trust_level})"
            )

    def add_violation(self, violation_type: ViolationType, details: Dict[str, Any], severity: int = 5):
        """Add a violation to the client's record"""
        violation = ClientViolation(
            violation_type=violation_type,
            timestamp=time.time(),
            details=details,
            severity=severity
        )

        self.violations.append(violation)

        # Keep only recent violations (memory protection)
        if len(self.violations) > 20:
            self.violations = self.violations[-20:]

        # Adjust reputation based on violation severity
        reputation_penalty = severity * 2
        self.reputation_score = max(0, self.reputation_score - reputation_penalty)

        # Auto-block for severe violations
        if severity >= 8 or len(self.violations) >= 5:
            self.block_client(f"Severe violations: {violation_type.value}")

    def block_client(self, reason: str, duration_seconds: int = 900):  # 15 minutes default
        """Block client for specified duration"""
        self.is_blocked = True
        self.block_until = time.time() + duration_seconds
        self.block_reason = reason
        self.client_type = ClientType.BLOCKED

        logger.warning(f"Blocked client {self.client_id[:8]}: {reason} (duration: {duration_seconds}s)")

    def unblock_client(self):
        """Unblock client if block period has expired"""
        if self.is_blocked and time.time() >= self.block_until:
            self.is_blocked = False
            self.block_until = 0
            self.block_reason = ""

            # Reset to previous classification
            if self.reputation_score >= 60:
                self.client_type = ClientType.VIP
            elif self.reputation_score >= 40:
                self.client_type = ClientType.HUMAN
            else:
                self.client_type = ClientType.SUSPICIOUS

            logger.info(f"Unblocked client {self.client_id[:8]}")
            return True

        return False

    def get_rate_limits(self, base_limit: int, base_burst: int) -> Tuple[int, int]:
        """Get adaptive rate limits based on client profile"""
        # Base multipliers by client type
        type_multipliers = {
            ClientType.VIP: (2.0, 2.0),
            ClientType.HUMAN: (1.0, 1.0),
            ClientType.UNKNOWN: (0.8, 0.8),
            ClientType.BOT_FRIENDLY: (0.5, 0.3),
            ClientType.BOT_AGGRESSIVE: (0.2, 0.1),
            ClientType.SUSPICIOUS: (0.3, 0.2),
            ClientType.BLOCKED: (0.0, 0.0)
        }

        rate_mult, burst_mult = type_multipliers.get(self.client_type, (0.5, 0.5))

        # Adjust based on trust level
        trust_bonus = (self.trust_level - 1) * 0.1  # 0% to 40% bonus
        rate_mult += trust_bonus
        burst_mult += trust_bonus

        # Calculate final limits
        final_rate = max(1, int(base_limit * rate_mult))
        final_burst = max(1, int(base_burst * burst_mult))

        return final_rate, final_burst

class UltimateRateLimiter:
    """
    Ultimate Advanced Rate Limiter v3.0

    FEATURES:
    - Intelligent client profiling and classification
    - Adaptive rate limits based on behavior analysis
    - Memory leak prevention with automatic cleanup
    - Advanced bot detection and blocking
    - VIP recognition and preferential treatment
    - Comprehensive violation tracking and scoring
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 30,
        window_minutes: int = 1,
        max_clients: int = 10000,
        cleanup_interval: int = 300,  # 5 minutes
        block_duration: int = 900     # 15 minutes
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_seconds = window_minutes * 60
        self.max_clients = max_clients
        self.cleanup_interval = cleanup_interval
        self.default_block_duration = block_duration

        # Client storage with memory protection
        self.clients: Dict[str, ClientProfile] = {}
        self._lock = asyncio.Lock()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Global statistics
        self.global_stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "total_clients": 0,
            "blocked_clients": 0,
            "start_time": time.time()
        }

        # Start background tasks
        self._start_background_tasks()

        logger.info(
            f"Advanced rate limiter initialized: "
            f"{requests_per_minute}/min, burst: {burst_limit}, max_clients: {max_clients}"
        )

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())

    async def _cleanup_loop(self):
        """Background cleanup of inactive clients"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_inactive_clients()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _analysis_loop(self):
        """Background behavior analysis and optimization"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._analyze_global_patterns()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")

    async def _cleanup_inactive_clients(self):
        """Remove inactive clients to prevent memory explosion"""
        async with self._lock:
            current_time = time.time()
            inactive_threshold = 3600  # 1 hour
            clients_to_remove = []

            for client_id, profile in list(self.clients.items()):
                # Remove if inactive and not important
                if (current_time - profile.last_seen > inactive_threshold and
                    profile.client_type not in [ClientType.VIP, ClientType.BLOCKED]):
                    clients_to_remove.append(client_id)

            # Emergency cleanup if at memory limit
            if len(self.clients) > self.max_clients:
                # Sort by last seen and remove oldest non-VIP clients
                sorted_clients = sorted(
                    [(k, v) for k, v in self.clients.items() if v.client_type != ClientType.VIP],
                    key=lambda x: x[1].last_seen
                )

                excess_count = len(self.clients) - self.max_clients
                for client_id, _ in sorted_clients[:excess_count]:
                    clients_to_remove.append(client_id)

            # Remove clients
            for client_id in clients_to_remove:
                del self.clients[client_id]

            if clients_to_remove:
                logger.debug(f"Cleaned up {len(clients_to_remove)} inactive clients")

    async def _analyze_global_patterns(self):
        """Analyze global patterns for threat detection"""
        async with self._lock:
            current_time = time.time()

            # Unblock clients whose block period has expired
            unblocked_count = 0
            for profile in self.clients.values():
                if profile.unblock_client():
                    unblocked_count += 1

            if unblocked_count > 0:
                logger.info(f"Automatically unblocked {unblocked_count} clients")

            # Update global statistics
            total_blocked = sum(1 for p in self.clients.values() if p.is_blocked)
            self.global_stats.update({
                "total_clients": len(self.clients),
                "blocked_clients": total_blocked,
                "vip_clients": sum(1 for p in self.clients.values() if p.client_type == ClientType.VIP),
                "bot_clients": sum(1 for p in self.clients.values() 
                                 if p.client_type in [ClientType.BOT_FRIENDLY, ClientType.BOT_AGGRESSIVE])
            })

    def _get_client_profile(self, client_id: str) -> ClientProfile:
        """Get or create client profile with memory protection"""
        if client_id not in self.clients:
            # Memory protection: remove oldest client if at limit
            if len(self.clients) >= self.max_clients:
                oldest_client = min(
                    self.clients.values(),
                    key=lambda c: c.last_seen if c.client_type != ClientType.VIP else float('inf')
                )
                if oldest_client.client_type != ClientType.VIP:
                    del self.clients[oldest_client.client_id]

            self.clients[client_id] = ClientProfile(client_id)

        return self.clients[client_id]

    async def check_rate_limit(
        self,
        client_id: str,
        endpoint: str = "default",
        user_agent: str = "unknown",
        country: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """
        Comprehensive rate limit check with intelligent profiling

        Args:
            client_id: Unique client identifier (usually IP address)
            endpoint: API endpoint being accessed
            user_agent: Client user agent string
            country: Client country (if available)
            additional_context: Additional context for analysis

        Returns:
            RateLimitResult with detailed information
        """
        start_time = time.time()
        current_time = start_time

        async with self._lock:
            self.global_stats["total_requests"] += 1

            # Get client profile
            profile = self._get_client_profile(client_id)

            # Check if client is blocked
            if profile.is_blocked:
                if current_time < profile.block_until:
                    remaining_block = int(profile.block_until - current_time)
                    self.global_stats["blocked_requests"] += 1

                    return RateLimitResult(
                        allowed=False,
                        current_count=0,
                        limit=0,
                        retry_after=remaining_block,
                        message=f"Client blocked: {profile.block_reason}",
                        client_type=ClientType.BLOCKED,
                        violation_type=ViolationType.SECURITY_VIOLATION,
                        additional_info={
                            "block_reason": profile.block_reason,
                            "block_remaining": remaining_block
                        }
                    )
                else:
                    # Unblock if period expired
                    profile.unblock_client()

            # Get adaptive rate limits
            rate_limit, burst_limit = profile.get_rate_limits(
                self.requests_per_minute, self.burst_limit
            )

            # Clean old requests from sliding window
            cutoff_time = current_time - self.window_seconds
            while profile.request_history and profile.request_history[0] < cutoff_time:
                profile.request_history.popleft()

            # Check rate limits
            current_count = len(profile.request_history)

            # Check burst limit (last 60 seconds)
            burst_cutoff = current_time - 60
            recent_requests = sum(
                1 for req_time in profile.request_history
                if req_time >= burst_cutoff
            )

            # Determine if request should be allowed
            violation_type = None
            allowed = True

            if recent_requests >= burst_limit:
                allowed = False
                violation_type = ViolationType.BURST_LIMIT
            elif current_count >= rate_limit:
                allowed = False
                violation_type = ViolationType.RATE_LIMIT

            # Additional security checks
            if allowed and additional_context:
                # Check for suspicious patterns in request
                if self._detect_suspicious_patterns(additional_context):
                    allowed = False
                    violation_type = ViolationType.SUSPICIOUS_PATTERN

            if allowed:
                # Allow request
                profile.request_history.append(current_time)
                profile.update_activity(endpoint, user_agent, True, 0.0, country)

                return RateLimitResult(
                    allowed=True,
                    current_count=current_count + 1,
                    limit=rate_limit,
                    retry_after=0,
                    message="Request allowed",
                    client_type=profile.client_type,
                    additional_info={
                        "reputation_score": profile.reputation_score,
                        "trust_level": profile.trust_level,
                        "burst_used": recent_requests,
                        "burst_limit": burst_limit
                    }
                )
            else:
                # Block request and record violation
                self.global_stats["blocked_requests"] += 1

                # Calculate retry after
                if profile.request_history:
                    oldest_request = profile.request_history[0]
                    retry_after = int(self.window_seconds - (current_time - oldest_request)) + 1
                else:
                    retry_after = 60

                # Record violation
                violation_details = {
                    "endpoint": endpoint,
                    "current_count": current_count,
                    "limit": rate_limit,
                    "burst_used": recent_requests,
                    "burst_limit": burst_limit
                }

                severity = 3  # Default severity
                if violation_type == ViolationType.BURST_LIMIT:
                    severity = 5
                elif violation_type == ViolationType.SUSPICIOUS_PATTERN:
                    severity = 7

                profile.add_violation(violation_type, violation_details, severity)
                profile.update_activity(endpoint, user_agent, False, 0.0, country)

                # Create appropriate message
                if violation_type == ViolationType.BURST_LIMIT:
                    message = f"Burst limit exceeded ({recent_requests}/{burst_limit})"
                elif violation_type == ViolationType.RATE_LIMIT:
                    message = f"Rate limit exceeded ({current_count}/{rate_limit})"
                else:
                    message = f"Request blocked: {violation_type.value}"

                return RateLimitResult(
                    allowed=False,
                    current_count=current_count,
                    limit=rate_limit,
                    retry_after=retry_after,
                    message=message,
                    client_type=profile.client_type,
                    violation_type=violation_type,
                    additional_info={
                        "reputation_score": profile.reputation_score,
                        "trust_level": profile.trust_level,
                        "violation_count": len(profile.violations)
                    }
                )

    def _detect_suspicious_patterns(self, context: Dict[str, Any]) -> bool:
        """Detect suspicious patterns in request context"""
        # Check for SQL injection patterns
        if "query_text" in context:
            query = context["query_text"].lower()
            sql_patterns = [
                "union select", "drop table", "delete from", "insert into",
                "update set", "exec(", "execute(", "sp_", "xp_"
            ]
            if any(pattern in query for pattern in sql_patterns):
                return True

        # Check for XSS patterns
        if any(key.startswith("input_") for key in context):
            for key, value in context.items():
                if key.startswith("input_") and isinstance(value, str):
                    xss_patterns = ["<script", "javascript:", "onload=", "onerror="]
                    if any(pattern in value.lower() for pattern in xss_patterns):
                        return True

        return False

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive rate limiter statistics"""
        uptime = time.time() - self.global_stats["start_time"]

        # Client type distribution
        type_distribution = defaultdict(int)
        trust_distribution = defaultdict(int)

        for profile in self.clients.values():
            type_distribution[profile.client_type.value] += 1
            trust_distribution[f"trust_{profile.trust_level}"] += 1

        # Recent activity (last hour)
        recent_cutoff = time.time() - 3600
        recent_activity = sum(
            1 for profile in self.clients.values()
            if profile.last_seen >= recent_cutoff
        )

        return {
            "global_stats": {
                **self.global_stats,
                "uptime_seconds": round(uptime, 2),
                "requests_per_second": round(self.global_stats["total_requests"] / uptime, 2),
                "block_rate_percent": round(
                    (self.global_stats["blocked_requests"] / max(self.global_stats["total_requests"], 1)) * 100, 2
                )
            },
            "client_stats": {
                "total_clients": len(self.clients),
                "active_clients_last_hour": recent_activity,
                "max_clients": self.max_clients,
                "memory_usage_percent": round((len(self.clients) / self.max_clients) * 100, 2)
            },
            "client_type_distribution": dict(type_distribution),
            "trust_level_distribution": dict(trust_distribution),
            "configuration": {
                "base_rate_limit": self.requests_per_minute,
                "base_burst_limit": self.burst_limit,
                "window_seconds": self.window_seconds,
                "cleanup_interval": self.cleanup_interval
            }
        }

    async def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific client"""
        async with self._lock:
            profile = self.clients.get(client_id)
            if not profile:
                return None

            rate_limit, burst_limit = profile.get_rate_limits(
                self.requests_per_minute, self.burst_limit
            )

            return {
                "client_id": client_id,
                "client_type": profile.client_type.value,
                "reputation_score": profile.reputation_score,
                "trust_level": profile.trust_level,
                "total_requests": profile.total_requests,
                "error_rate": round(profile.error_rate * 100, 2),
                "is_blocked": profile.is_blocked,
                "block_reason": profile.block_reason if profile.is_blocked else None,
                "adaptive_limits": {
                    "rate_limit": rate_limit,
                    "burst_limit": burst_limit
                },
                "recent_violations": len([
                    v for v in profile.violations
                    if time.time() - v.timestamp < 3600  # Last hour
                ]),
                "user_agents": list(profile.user_agents),
                "endpoints_accessed": list(profile.endpoints_accessed),
                "countries": list(profile.countries),
                "created_at": datetime.fromtimestamp(profile.created_at).isoformat(),
                "last_seen": datetime.fromtimestamp(profile.last_seen).isoformat()
            }

    async def cleanup(self):
        """Cleanup rate limiter resources"""
        logger.info("🧹 Starting rate limiter cleanup...")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        # Clear client data
        async with self._lock:
            self.clients.clear()

        logger.info("✅ Rate limiter cleanup completed")
