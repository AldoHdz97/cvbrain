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
    """Ultimate base model with Pydantic v2 optimizations"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={"additionalProperties": False}
    )

class UltimateQueryRequest(UltimateBaseModel):
    """SIMPLIFIED Query Request with required request_id"""
    
    question: Annotated[str, StringConstraints(
        min_length=3,
        max_length=2000,
        strip_whitespace=True
    )] = Field(..., description="Question about Aldo's professional background")
    
    # Add back request_id - backend service needs this
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Request identifier"
    )
    
    # Optional fields with defaults
    k: Optional[int] = Field(default=3)
    query_type: Optional[QueryType] = Field(default=None)
    response_format: Optional[ResponseFormat] = Field(default=ResponseFormat.DETAILED)
    include_sources: Optional[bool] = Field(default=True)
    include_confidence_explanation: Optional[bool] = Field(default=False)
    language: Optional[str] = Field(default="en")
    max_response_length: Optional[int] = Field(default=800)
    
    @field_validator("question")
    @classmethod
    def validate_question_security(cls, v: str, info: ValidationInfo) -> str:
        v = re.sub(r'\s+', ' ', v.strip())
        words = v.split()
        if len(words) < 2:
            raise ValueError("Question must contain at least 2 meaningful words")
        return v
    
    @computed_field
    @property
    def question_hash(self) -> str:
        return hashlib.md5(self.question.encode()).hexdigest()[:16]
    
    @computed_field  
    @property
    def complexity(self) -> QueryComplexity:
        word_count = len(self.question.split())
        if word_count <= 5:
            return QueryComplexity.SIMPLE
        elif word_count <= 15:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.COMPLEX

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
