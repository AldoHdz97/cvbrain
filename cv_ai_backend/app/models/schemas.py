"""
CV-AI Backend Data Models
Simplified and clean Pydantic models
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import uuid4
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

class QueryType(str, Enum):
    """Query type classification"""
    SKILLS = "skills"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    PROJECTS = "projects"
    SUMMARY = "summary"
    CONTACT = "contact"
    GENERAL = "general"

class ResponseFormat(str, Enum):
    """Response format options"""
    DETAILED = "detailed"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"
    CONVERSATIONAL = "conversational"

class QueryRequest(BaseModel):
    """CV query request"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    question: str = Field(..., min_length=3, max_length=1000, description="Question about professional background")
    k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")
    query_type: Optional[QueryType] = None
    response_format: ResponseFormat = ResponseFormat.DETAILED
    include_sources: bool = True
    max_response_length: int = Field(default=800, ge=100, le=2000)

class QueryResponse(BaseModel):
    """CV query response"""
    
    # Core response
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    answer: str = Field(..., description="AI-generated response")
    
    # Query info
    query_type: QueryType
    
    # Context info
    relevant_chunks: int = Field(ge=0, le=10)
    similarity_scores: List[float] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    
    # Performance
    processing_time: float = Field(ge=0.0)
    model_used: str
    cache_hit: bool = False
    
    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., pattern="^(healthy|unhealthy)$")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str
    components: Dict[str, str] = Field(default_factory=dict)

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
