"""
FastAPI Server for CV-AI Backend - WITH INTERVIEW SCHEDULING SYSTEM
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uuid
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.schemas import UltimateQueryRequest, UltimateQueryResponse
from app.services.ultimate_cv_service import get_ultimate_cv_service, cleanup_ultimate_cv_service

# ===================================================================
# 🆕 NEW IMPORTS FOR INTERVIEW SCHEDULING
# ===================================================================
from app.database.database import get_database_manager, get_db_session, cleanup_database
from app.models.interview_models import InterviewRequest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================================
# 🆕 PYDANTIC MODELS FOR INTERVIEW SCHEDULING
# ===================================================================

class InterviewScheduleRequest(BaseModel):
    """Request model for scheduling interviews"""
    selected_day: str = Field(..., description="Selected interview day (e.g., 'Monday, January 15')")
    selected_time: str = Field(..., description="Selected time slot (e.g., '10:00 AM - 11:00 AM')")
    contact_info: str = Field(..., min_length=10, max_length=1000, description="Contact information (email, phone, etc.)")

class InterviewScheduleResponse(BaseModel):
    """Response model for interview scheduling"""
    success: bool
    message: str
    interview_id: Optional[str] = None
    scheduled_day: Optional[str] = None
    scheduled_time: Optional[str] = None
    status: str = "pending"

class InterviewListResponse(BaseModel):
    """Response model for listing interviews"""
    interviews: List[dict]
    total_count: int
    pending_count: int
    metadata: dict

# ===================================================================
# LIFESPAN MANAGEMENT - UPDATED WITH DATABASE
# ===================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management - NOW WITH DATABASE"""
    # Startup
    logger.info("🚀 Starting CV-AI Backend with Interview Scheduling...")
    try:
        # Initialize CV Service
        service = await get_ultimate_cv_service()
        logger.info("✅ CV-AI Service initialized")
        
        # 🆕 Initialize Database
        db_manager = await get_database_manager()
        logger.info("✅ Interview Database initialized")
        
        yield
        
    finally:
        # Shutdown
        logger.info("🧹 Shutting down CV-AI Backend...")
        await cleanup_ultimate_cv_service()
        await cleanup_database()
        logger.info("✅ Cleanup completed")

# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="CV-AI Backend Ultimate - WITH INTERVIEW SCHEDULING",
    description="AI-powered CV query system with interview scheduling capabilities",
    version=settings.app_version,
    lifespan=lifespan
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.cors_origins],
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
        max_age=settings.cors_max_age,
    )

@app.get("/")
async def root():
    """Root endpoint - UPDATED"""
    return {
        "message": "CV-AI Backend Ultimate v3.0 - WITH INTERVIEW SCHEDULING", 
        "status": "operational",
        "docs": "/docs",
        "features": [
            "conversational_memory", 
            "multilingual", 
            "advanced_caching",
            "interview_scheduling",  # 🆕 NEW FEATURE
            "sqlite_database"        # 🆕 NEW FEATURE
        ]
    }

# ===================================================================
# 🔧 EXISTING CV QUERY ENDPOINTS (UNCHANGED)
# ===================================================================

@app.post("/query", response_model=UltimateQueryResponse)
async def query_cv(
    request: UltimateQueryRequest,
    session_id: Optional[str] = None,
    maintain_context: Optional[bool] = True
):
    """
    🔥 FIXED: Query CV with AI-powered semantic search and conversational memory
    
    Now supports:
    - Conversational memory across questions
    - Session-based context management  
    - Automatic language detection
    - Natural follow-up questions
    """
    try:
        # 🔧 GENERAR SESSION_ID SI NO ESTÁ EN REQUEST O PARÁMETRO
        effective_session_id = None
        
        # Prioridad: parámetro URL > campo en request > generar nuevo
        if session_id:
            effective_session_id = session_id
        elif hasattr(request, 'session_id') and request.session_id:
            effective_session_id = request.session_id
        else:
            effective_session_id = str(uuid.uuid4())
            logger.info(f"🆕 Generated new session_id: {effective_session_id}")
        
        # 🔧 DETERMINAR MAINTAIN_CONTEXT
        effective_maintain_context = maintain_context
        if hasattr(request, 'maintain_context') and request.maintain_context is not None:
            effective_maintain_context = request.maintain_context
        
        logger.info(f"🗣️  Conversational query - Session: {effective_session_id}, Context: {effective_maintain_context}")
        
        # 🔧 USAR SERVICIO CONVERSACIONAL
        service = await get_ultimate_cv_service()
        response = await service.query_cv_conversational(
            request=request,
            session_id=effective_session_id,
            maintain_context=effective_maintain_context
        )
        
        # 🔧 AGREGAR SESSION_ID A METADATA PARA EL FRONTEND
        if not response.metadata:
            response.metadata = {}
        response.metadata["session_id"] = effective_session_id
        response.metadata["maintain_context"] = effective_maintain_context
        
        return response
        
    except Exception as e:
        logger.error(f"Conversational query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-conversational")
async def query_conversational(
    question: str,
    session_id: Optional[str] = None,
    maintain_context: bool = True,
    k: int = 5
):
    """
    🆕 Endpoint simplificado específico para conversaciones
    
    Parámetros:
    - question: La pregunta a hacer
    - session_id: ID de sesión para mantener contexto (se genera automáticamente si no se proporciona)
    - maintain_context: Si mantener el contexto conversacional
    - k: Número de chunks de contexto a recuperar
    """
    try:
        # Generar session_id si no se proporciona
        if not session_id:
            session_id = str(uuid.uuid4())
        
        logger.info(f"🗣️  Simple conversational query - Session: {session_id}")
        
        # Crear request
        cv_request = UltimateQueryRequest(
            question=question,
            k=k
        )
        
        # Obtener servicio y hacer query conversacional
        service = await get_ultimate_cv_service()
        response = await service.query_cv_conversational(
            request=cv_request,
            session_id=session_id,
            maintain_context=maintain_context
        )
        
        # Respuesta simplificada
        return {
            "answer": response.answer,
            "session_id": session_id,
            "conversation_turn": response.metadata.get("conversation_turn", 1),
            "confidence_score": response.confidence_score,
            "processing_time": response.processing_metrics["total_time_seconds"],
            "language_detected": response.language
        }
        
    except Exception as e:
        logger.error(f"Simple conversational query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
# 🆕 NEW INTERVIEW SCHEDULING ENDPOINTS
# ===================================================================

@app.post("/schedule-interview", response_model=InterviewScheduleResponse)
async def schedule_interview(
    request: InterviewScheduleRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db_session)
):
    """
    🆕 Schedule a new interview appointment
    
    This endpoint allows users to schedule interview appointments by providing:
    - Selected day and time
    - Contact information
    
    The system automatically stores the request in SQLite database for review.
    """
    try:
        logger.info(f"📅 New interview scheduling request: {request.selected_day} at {request.selected_time}")
        
        # Extract client metadata
        client_ip = http_request.client.host if http_request.client else "unknown"
        user_agent = http_request.headers.get("user-agent", "unknown")
        
        # Create interview request in database
        interview = InterviewRequest.from_request_data(
            selected_day=request.selected_day,
            selected_time=request.selected_time,
            contact_info=request.contact_info,
            user_ip=client_ip,
            user_agent=user_agent
        )
        
        # Save to database
        db.add(interview)
        await db.commit()
        await db.refresh(interview)
        
        logger.info(f"✅ Interview scheduled successfully: ID {interview.id}")
        
        return InterviewScheduleResponse(
            success=True,
            message="Interview scheduled successfully! You will be contacted soon.",
            interview_id=interview.id,
            scheduled_day=request.selected_day,
            scheduled_time=request.selected_time,
            status="pending"
        )
        
    except Exception as e:
        logger.error(f"❌ Interview scheduling failed: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to schedule interview: {str(e)}"
        )

@app.get("/admin/interviews", response_model=InterviewListResponse)
async def list_interviews(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session)
):
    """
    🆕 Admin endpoint to list all interview requests
    
    Query parameters:
    - status: Filter by status (pending, confirmed, cancelled, completed)
    - limit: Maximum number of results (default: 50)
    - offset: Number of results to skip for pagination (default: 0)
    """
    try:
        logger.info(f"📋 Admin fetching interviews: status={status}, limit={limit}, offset={offset}")
        
        # Build query
        query = select(InterviewRequest)
        
        if status:
            query = query.where(InterviewRequest.status == status)
        
        # Add ordering and pagination
        query = query.order_by(InterviewRequest.created_at.desc()).offset(offset).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        interviews = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(InterviewRequest.id))
        if status:
            count_query = count_query.where(InterviewRequest.status == status)
        
        total_result = await db.execute(count_query)
        total_count = total_result.scalar() or 0
        
        # Get pending count
        pending_result = await db.execute(
            select(func.count(InterviewRequest.id)).where(InterviewRequest.status == "pending")
        )
        pending_count = pending_result.scalar() or 0
        
        # Convert to dict format
        interviews_data = [interview.to_dict() for interview in interviews]
        
        logger.info(f"✅ Retrieved {len(interviews_data)} interviews (total: {total_count})")
        
        return InterviewListResponse(
            interviews=interviews_data,
            total_count=total_count,
            pending_count=pending_count,
            metadata={
                "status_filter": status,
                "limit": limit,
                "offset": offset,
                "returned_count": len(interviews_data)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch interviews: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch interviews: {str(e)}"
        )

@app.get("/admin/interviews/{interview_id}")
async def get_interview_details(
    interview_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """
    🆕 Get detailed information about a specific interview
    """
    try:
        # Fetch interview by ID
        result = await db.execute(
            select(InterviewRequest).where(InterviewRequest.id == interview_id)
        )
        interview = result.scalar_one_or_none()
        
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        return interview.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to fetch interview {interview_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch interview: {str(e)}"
        )

@app.put("/admin/interviews/{interview_id}/status")
async def update_interview_status(
    interview_id: str,
    new_status: str,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    🆕 Update interview status (admin only)
    
    Valid statuses: pending, confirmed, cancelled, completed
    """
    try:
        valid_statuses = ["pending", "confirmed", "cancelled", "completed"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}"
            )
        
        # Fetch interview
        result = await db.execute(
            select(InterviewRequest).where(InterviewRequest.id == interview_id)
        )
        interview = result.scalar_one_or_none()
        
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        # Update status
        interview.status = new_status
        if notes:
            interview.notes = notes
        
        await db.commit()
        await db.refresh(interview)
        
        logger.info(f"✅ Interview {interview_id} status updated to: {new_status}")
        
        return {
            "success": True,
            "message": f"Interview status updated to: {new_status}",
            "interview": interview.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update interview status: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update interview: {str(e)}"
        )

# ===================================================================
# 🆕 DATABASE HEALTH CHECK ENDPOINTS
# ===================================================================

@app.get("/health-db")
async def database_health_check():
    """
    🆕 Check database health and statistics
    """
    try:
        db_manager = await get_database_manager()
        
        # Get database health
        is_healthy = await db_manager.health_check()
        
        # Get database statistics
        stats = await db_manager.get_stats()
        
        return {
            "database_healthy": is_healthy,
            "database_stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return {
            "database_healthy": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ===================================================================
# 🔧 EXISTING CONVERSATION MANAGEMENT ENDPOINTS (UNCHANGED)
# ===================================================================

@app.get("/conversation/{session_id}")
async def get_conversation_history(session_id: str):
    """Obtener historial de una conversación específica"""
    try:
        service = await get_ultimate_cv_service()
        history = await service.get_conversation_history(session_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Limpiar una conversación específica"""
    try:
        service = await get_ultimate_cv_service()
        cleared = await service.clear_conversation(session_id)
        return {"cleared": cleared, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/active")
async def get_active_conversations():
    """Obtener estadísticas de conversaciones activas"""
    try:
        service = await get_ultimate_cv_service()
        stats = await service.get_active_conversations()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
# 🔧 EXISTING SYSTEM ENDPOINTS (UPDATED)
# ===================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint - UPDATED WITH DATABASE"""
    try:
        service = await get_ultimate_cv_service()
        stats = await service.get_comprehensive_stats()
        
        # 🆕 Add database health check
        db_manager = await get_database_manager()
        db_healthy = await db_manager.health_check()
        db_stats = await db_manager.get_stats()
        
        return {
            "status": "healthy",
            "service_info": stats["service_info"],
            "components": {
                "connection_manager": "operational",
                "cache_system": "operational", 
                "chromadb_manager": "operational",
                "conversation_manager": "operational",
                "interview_database": "operational" if db_healthy else "degraded"  # 🆕 NEW
            },
            "features": {
                "conversational_memory": True,
                "multilingual_support": True,
                "automatic_language_detection": True,
                "interview_scheduling": True,  # 🆕 NEW
                "sqlite_database": True        # 🆕 NEW
            },
            "database_info": {              # 🆕 NEW SECTION
                "healthy": db_healthy,
                "stats": db_stats
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics - UPDATED"""
    try:
        service = await get_ultimate_cv_service()
        cv_stats = await service.get_comprehensive_stats()
        
        # 🆕 Add database statistics
        db_manager = await get_database_manager()
        db_stats = await db_manager.get_stats()
        
        # Combine statistics
        cv_stats["database_manager"] = db_stats
        
        return cv_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/setup-cv-data")
async def setup_cv_data():
    """One-time setup to load CV data into ChromaDB"""
    try:
        # Import your setup script
        from app.scripts.setup_cv_data import load_cv_data
        
        result = await load_cv_data()
        
        if result:
            return {
                "status": "success",
                "message": "CV data loaded successfully into ChromaDB",
                "action": "Your CV-AI system is now ready!"
            }
        else:
            return {
                "status": "error", 
                "message": "Failed to load CV data"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Setup failed: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers
    )
