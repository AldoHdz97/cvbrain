"""
CV-AI Backend Ultimate Data Models v3.0
Latest Pydantic v2 patterns with comprehensive validation and security

FEATURES:
- Latest Pydantic v2.11+ patterns and optimizations
- Enhanced security validation (prompt injection prevention)
- Performance-optimized serialization and validation
- Comprehensive error handling with detailed context
- Memory-efficient field validation with compiled regex
- Advanced computed fields and model composition
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Annotated, Literal
from uuid import UUID, uuid4
import re
import hashlib
from decimal import Decimal

from pydantic import (
    BaseModel, Field, field_validator, model_validator, computed_field, 
    ConfigDict, StringConstraints, ValidationInfo, field_serializer
)

# Performance optimization: Pre-compile regex patterns
SECURITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in [
        r'ignore\s+(?:previous\s+)?instructions',
        r'system\s*[:,]\s*',
        r'assistant\s*[:,]\s*',
        r'<\s*/?(?:system|assistant)\s*>',
        r'jailbreak\s*(?:mode|prompt)?',
        r'pretend\s+(?:you\s+)?are\s+(?:a\s+)?',
        r'act\s+as\s+(?:if\s+)?(?:you\s+)?(?:are\s+)?(?:a\s+)?',
        r'forget\s+(?:everything|all|your\s+instructions)',
        r'override\s+(?:previous\s+)?instructions',
        r'disregard\s+(?:previous\s+)?(?:all\s+)?instructions',
        r'new\s+instructions?\s*[:,]?',
        r'breaking?\s+character',
        # Latest 2024 patterns
        r'bypass\s+(?:all\s+)?(?:safety\s+)?(?:filters?|guards?)',
        r'unlock\s+(?:developer|admin|god)\s+mode',
        r'enable\s+(?:debug|unrestricted|dev)\s+mode',
        r'switch\s+to\s+(?:developer|admin)\s+mode',
        r'reveal\s+(?:system\s+)?(?:prompt|instructions)',
        r'show\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions)'
    ]
]

class QueryType(str, Enum):
    """Enhanced query types with comprehensive coverage"""
    # Core types
    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    PROJECTS = "projects"
    SUMMARY = "summary"

    # Extended types
    CONTACT = "contact"
    ACHIEVEMENTS = "achievements"
    CERTIFICATIONS = "certifications"
    LANGUAGES = "languages"
    TOOLS = "tools"
    METHODOLOGIES = "methodologies"
    TECHNICAL = "technical"  # ✅ ADD THIS LINE

    # Meta types
    GENERAL = "general"
    CLARIFICATION = "clarification"

class ConfidenceLevel(str, Enum):
    """Statistical confidence levels with clear thresholds"""
    VERY_HIGH = "very_high"    # >0.90 similarity
    HIGH = "high"             # 0.80-0.90 similarity
    MEDIUM = "medium"         # 0.65-0.80 similarity
    LOW = "low"              # 0.50-0.65 similarity
    VERY_LOW = "very_low"    # <0.50 similarity

class ResponseFormat(str, Enum):
    """Response format options for customization"""
    DETAILED = "detailed"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"

class QueryComplexity(str, Enum):
    """Query complexity levels for optimization"""
    SIMPLE = "simple"          # ≤5 words
    MEDIUM = "medium"          # 6-15 words
    COMPLEX = "complex"        # 16-30 words
    VERY_COMPLEX = "very_complex"  # >30 words

class UltimateBaseModel(BaseModel):
    """
    Ultimate base model with latest Pydantic v2 optimizations

    FEATURES:
    - Performance-optimized configuration
    - Memory-efficient validation
    - Enhanced JSON serialization
    - Consistent validation across all models
    """

    model_config = ConfigDict(
        # Performance optimizations (Pydantic v2.11+)
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_default=True,
        arbitrary_types_allowed=False,

        # Memory optimizations
        extra="forbid",  # Prevent memory bloat from unexpected fields
        frozen=False,    # Allow mutations for performance

        # Serialization optimizations
        ser_json_bytes= 'utf8',
        ser_json_timedelta="float",
        ser_json_inf_nan="constants",

        # JSON schema optimizations
        json_schema_extra={
            "additionalProperties": False
        }
    )

class UltimateQueryRequest(UltimateBaseModel):
    """
    Ultimate query request with comprehensive validation and security

    SECURITY FEATURES:
    - Advanced prompt injection prevention
    - Input sanitization and normalization
    - Rate limiting compatibility
    - Content analysis and classification
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What programming languages and frameworks do you have experience with?",
                "k": 3,
                "query_type": "skills",
                "response_format": "detailed",
                "include_sources": True,
                "include_confidence_explanation": False,
                "language": "en",
                "max_response_length": 800
            }
        }
    )

    # Core fields with enhanced validation
    question: Annotated[str, StringConstraints(
        min_length=3,
        max_length=2000,
        strip_whitespace=True
    )] = Field(
        ...,
        description="Natural language question about Aldo's professional background",
        examples=[
            "What are your main technical skills and programming languages?",
            "Tell me about your experience in data analysis and machine learning",
            "What notable projects have you worked on with Python and APIs?",
            "Describe your educational background and certifications",
            "What achievements and accomplishments are you most proud of?"
        ]
    )

    # Query parameters
    k: Annotated[int, Field(ge=1, le=10)] = Field(
        default=3,
        description="Number of relevant document chunks to retrieve for context"
    )

    query_type: Optional[QueryType] = Field(
        default=None,
        description="Query type for optimized processing (auto-detected if not provided)"
    )

    # Response customization
    response_format: ResponseFormat = Field(
        default=ResponseFormat.DETAILED,
        description="Desired response format and style"
    )

    include_sources: bool = Field(
        default=True,
        description="Include source document previews in response"
    )

    include_confidence_explanation: bool = Field(
        default=False,
        description="Include detailed confidence score explanation"
    )

    max_response_length: Optional[Annotated[int, Field(ge=100, le=2000)]] = Field(
        default=None,
        description="Maximum response length in characters"
    )

    # Localization
    language: Annotated[str, StringConstraints(
        pattern=r"^[a-z]{2}(?:-[A-Z]{2})?$"
    )] = Field(
        default="en",
        description="Response language (ISO 639-1 code, optionally with country)"
    )

    # Advanced options
    enable_streaming: bool = Field(
        default=False,
        description="Enable response streaming (if supported)"
    )

    temperature_override: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = Field(
        default=None,
        description="Override default AI model temperature"
    )

    # Request tracking and context
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier for tracking and correlation"
    )

    client_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional client context and metadata"
    )

    # Timestamp
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request submission timestamp"
    )

    @field_validator("question")
    @classmethod
    def validate_question_security(cls, v: str, info: ValidationInfo) -> str:
        """
        CRITICAL SECURITY: Comprehensive question validation with latest patterns

        FEATURES:
        - Advanced prompt injection detection
        - Content normalization and sanitization
        - Suspicious pattern detection
        - Character distribution analysis
        """

        # Normalize whitespace and basic cleanup
        v = re.sub(r'\s+', ' ', v.strip())

        # CRITICAL: Security validation using compiled patterns
        for pattern in SECURITY_PATTERNS:
            if pattern.search(v):
                raise ValueError(
                    f"Question contains potentially unsafe content that could compromise system security"
                )

        # Enhanced content validation
        words = v.split()
        if len(words) < 2:
            raise ValueError("Question must contain at least 2 meaningful words")

        # Check for suspicious patterns
        if re.search(r'(.){8,}', v):  # Repeated characters (8+ times)
            raise ValueError("Question contains suspicious repeated characters")

        # Advanced character distribution analysis
        total_chars = len(v)
        alpha_chars = len(re.findall(r'[a-zA-Z]', v))
        digit_chars = len(re.findall(r'[0-9]', v))
        space_chars = len(re.findall(r'\s', v))
        punct_chars = len(re.findall(r'[.!?,:;]', v))
        special_chars = total_chars - alpha_chars - digit_chars - space_chars - punct_chars

        # Validate character distribution
        if alpha_chars / total_chars < 0.5:  # Less than 50% alphabetic
            raise ValueError("Question must contain primarily alphabetic characters")

        if special_chars / total_chars > 0.15:  # More than 15% special characters
            raise ValueError("Question contains too many special characters")

        # Language detection (basic)
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError("Question must contain at least some Latin characters")

        return v

    @field_validator("k")
    @classmethod
    def validate_k_optimization(cls, v: int, info: ValidationInfo) -> int:
        """Optimize k parameter based on query context"""

        # Access other validated fields through info.data
        if hasattr(info, 'data') and info.data:
            query_type = info.data.get("query_type")
            question = info.data.get("question", "")

            # Auto-adjust k based on query type and complexity
            if query_type == QueryType.SUMMARY and v < 5:
                return 5  # Summary queries need more context
            elif query_type in [QueryType.CONTACT, QueryType.CERTIFICATIONS] and v > 3:
                return 3  # Simple queries need less context
            elif len(question.split()) > 20 and v < 5:  # Complex questions
                return min(v + 2, 8)  # Add more context but cap at 8

        return v

    @computed_field
    @property
    def question_hash(self) -> str:
        """Computed cache key for performance optimization"""
        cache_string = f"{self.question}:{self.k}:{self.query_type}:{self.response_format}:{self.language}"
        return hashlib.blake2b(cache_string.encode(), digest_size=16).hexdigest()

    @computed_field
    @property
    def complexity(self) -> QueryComplexity:
        """Computed query complexity for optimization"""
        word_count = len(self.question.split())

        if word_count <= 5:
            return QueryComplexity.SIMPLE
        elif word_count <= 15:
            return QueryComplexity.MEDIUM
        elif word_count <= 30:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.VERY_COMPLEX

    @computed_field
    @property
    def estimated_processing_time(self) -> float:
        """Estimated processing time based on complexity"""
        base_time = 1.0  # Base processing time in seconds

        complexity_multipliers = {
            QueryComplexity.SIMPLE: 0.8,
            QueryComplexity.MEDIUM: 1.0,
            QueryComplexity.COMPLEX: 1.3,
            QueryComplexity.VERY_COMPLEX: 1.6
        }

        return base_time * complexity_multipliers.get(self.complexity, 1.0) * self.k * 0.2

    @model_validator(mode='after')
    def validate_request_consistency(self) -> 'UltimateQueryRequest':
        """Comprehensive model validation for consistency"""

        # Adjust response format based on query type
        if self.query_type == QueryType.TECHNICAL and self.response_format == ResponseFormat.CONVERSATIONAL:
            self.response_format = ResponseFormat.TECHNICAL

        # Validate max_response_length compatibility
        if self.response_format == ResponseFormat.SUMMARY and self.max_response_length and self.max_response_length > 400:
            self.max_response_length = 400
        elif self.response_format == ResponseFormat.DETAILED and self.max_response_length and self.max_response_length < 200:
            self.max_response_length = 200

        return self

class UltimateQueryResponse(UltimateBaseModel):
    """
    Ultimate query response with comprehensive metadata and performance tracking

    FEATURES:
    - Detailed processing metrics and performance data
    - Multi-factor confidence scoring with explanations
    - Comprehensive source attribution and metadata
    - Quality assessment and optimization suggestions
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "answer": "I have extensive experience with multiple programming languages and frameworks...",
                "query_type": "skills",
                "confidence_level": "high",
                "confidence_score": 0.847,
                "confidence_factors": {
                    "semantic_similarity": 0.892,
                    "source_diversity": 0.833,
                    "content_completeness": 0.815
                },
                "relevant_chunks": 3,
                "similarity_scores": [0.892, 0.847, 0.823],
                "processing_metrics": {
                    "total_time_seconds": 2.34,
                    "embedding_time": 0.12,
                    "search_time": 0.08,
                    "generation_time": 2.01,
                    "cache_hit": False
                },
                "sources": ["Technical skills: Python, JavaScript, SQL...", "Project experience with..."],
                "model_used": "gpt-4o-mini",
                "timestamp": "2024-12-07T14:30:45.123Z"
            }
        }
    )

    # Core response data
    request_id: str = Field(..., description="Unique request identifier")
    answer: str = Field(..., description="AI-generated response based on CV content")

    # Query classification and analysis
    query_type: QueryType = Field(..., description="Detected or specified query type")
    query_complexity: QueryComplexity = Field(..., description="Analyzed query complexity")

    # Confidence scoring (multi-factor)
    confidence_level: ConfidenceLevel = Field(..., description="Overall confidence assessment")
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ..., description="Numerical confidence score (0.0-1.0)"
    )
    confidence_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of confidence calculation factors"
    )
    confidence_explanation: Optional[str] = Field(
        default=None,
        description="Human-readable confidence explanation"
    )

    # Source information and retrieval
    relevant_chunks: Annotated[int, Field(ge=0, le=20)] = Field(
        ..., description="Number of document chunks used for context"
    )
    similarity_scores: List[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        ..., description="Semantic similarity scores for each chunk"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Preview of source content used"
    )
    source_metadata: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed metadata for each source"
    )

    # Processing performance metrics
    processing_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed processing performance data"
    )

    # Model and generation info
    model_used: str = Field(..., description="AI model used for generation")
    model_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters used for generation"
    )
    tokens_used: Optional[int] = Field(
        default=None,
        description="Total tokens consumed (prompt + completion)"
    )

    # Response metadata
    response_format: ResponseFormat = Field(..., description="Response format used")
    language: str = Field(default="en", description="Response language")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response generation timestamp"
    )

    # Quality metrics
    quality_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response quality assessment metrics"
    )

    # Performance tracking
    cache_hit: bool = Field(default=False, description="Whether response used cached data")
    cache_key: Optional[str] = Field(default=None, description="Cache key used")

    # Additional context
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing metadata and context"
    )

    @field_validator("similarity_scores")
    @classmethod
    def validate_similarity_scores(cls, v: List[float], info: ValidationInfo) -> List[float]:
        """Validate similarity scores consistency"""
        if hasattr(info, 'data') and info.data:
            relevant_chunks = info.data.get("relevant_chunks", 0)
            if len(v) != relevant_chunks:
                raise ValueError("Number of similarity scores must match relevant_chunks")

        # Ensure scores are properly bounded and rounded
        return [round(max(0.0, min(1.0, score)), 4) for score in v]

    @field_validator("confidence_factors")
    @classmethod
    def validate_confidence_factors(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate confidence factor values"""
        for factor, value in v.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Confidence factor '{factor}' must be between 0.0 and 1.0")
        return {k: round(v, 4) for k, v in v.items()}

    @computed_field
    @property
    def overall_quality_score(self) -> float:
        """Computed overall quality assessment"""
        factors = []

        # Confidence factor (35% weight)
        factors.append(self.confidence_score * 0.35)

        # Source diversity factor (25% weight)
        if self.relevant_chunks > 0:
            diversity = min(self.relevant_chunks / 5.0, 1.0)
            factors.append(diversity * 0.25)

        # Response completeness factor (25% weight)
        answer_length = len(self.answer)
        if 150 <= answer_length <= 1000:
            completeness = 1.0
        elif answer_length > 0:
            # Penalize very short or very long responses
            optimal_length = 500
            completeness = 1.0 - abs(answer_length - optimal_length) / optimal_length
            completeness = max(0.0, min(1.0, completeness))
        else:
            completeness = 0.0
        factors.append(completeness * 0.25)

        # Performance factor (15% weight)
        processing_time = self.processing_metrics.get("total_time_seconds", 0)
        if processing_time <= 2.0:
            performance = 1.0
        elif processing_time <= 5.0:
            performance = 0.8
        elif processing_time <= 10.0:
            performance = 0.6
        else:
            performance = 0.4
        factors.append(performance * 0.15)

        return round(sum(factors), 4)

    @computed_field
    @property
    def performance_grade(self) -> str:
        """Human-readable performance grade"""
        score = self.overall_quality_score

        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Needs Improvement"

    @field_serializer("generated_at")
    def serialize_datetime(self, value: datetime) -> str:
        """Custom datetime serialization"""
        return value.isoformat()

# Health and system models
class HealthStatus(str, Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

class ComponentHealth(UltimateBaseModel):
    """Individual component health information"""
    status: HealthStatus
    message: str
    response_time_ms: Optional[float] = None
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    additional_info: Dict[str, Any] = Field(default_factory=dict)
    error_count: int = Field(default=0, ge=0)
    uptime_seconds: Optional[float] = None

class UltimateHealthResponse(UltimateBaseModel):
    """Comprehensive system health response"""
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    system_info: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Service information
    service_version: str
    environment: str
    uptime_seconds: float

    @computed_field
    @property
    def health_score(self) -> float:
        """Computed overall health score"""
        if not self.components:
            return 0.0

        healthy_count = sum(
            1 for comp in self.components.values() 
            if comp.status == HealthStatus.HEALTHY
        )

        return round(healthy_count / len(self.components), 3)

# Error handling models
class ErrorDetail(UltimateBaseModel):
    """Detailed error information"""
    type: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Internal error code")
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")

class UltimateErrorResponse(UltimateBaseModel):
    """Comprehensive error response"""
    error: ErrorDetail
    request_id: Optional[str] = None
    suggestion: Optional[str] = None
    support_info: Optional[Dict[str, str]] = None
    retry_after: Optional[int] = Field(None, description="Retry delay in seconds")

# Export all models
__all__ = [
    "QueryType", "ConfidenceLevel", "ResponseFormat", "QueryComplexity",
    "UltimateBaseModel", "UltimateQueryRequest", "UltimateQueryResponse",
    "HealthStatus", "ComponentHealth", "UltimateHealthResponse",
    "ErrorDetail", "UltimateErrorResponse"
]
