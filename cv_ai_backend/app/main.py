"""
FastAPI Server for CV-AI Backend - FIXED with Conversational Memory
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uuid
from typing import Optional

from app.core.config import get_settings
from app.models.schemas import UltimateQueryRequest, UltimateQueryResponse
from app.services.ultimate_cv_service import get_ultimate_cv_service, cleanup_ultimate_cv_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting CV-AI Backend...")
    try:
        service = await get_ultimate_cv_service()
        logger.info("✅ CV-AI Service initialized")
        yield
    finally:
        # Shutdown
        logger.info("🧹 Shutting down CV-AI Backend...")
        await cleanup_ultimate_cv_service()
        logger.info("✅ Cleanup completed")

# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="CV-AI Backend Ultimate",
    description="AI-powered CV query system with advanced caching and monitoring",
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
    """Root endpoint"""
    return {
        "message": "CV-AI Backend Ultimate v3.0 - Conversational Edition", 
        "status": "operational",
        "docs": "/docs",
        "features": ["conversational_memory", "multilingual", "advanced_caching"]
    }

# ===================================================================
# 🔧 ENDPOINT QUERY CORREGIDO CON MEMORIA CONVERSACIONAL
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

# ===================================================================
# 🆕 NUEVO ENDPOINT ESPECÍFICO PARA CONVERSACIONES
# ===================================================================

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
# 🆕 ENDPOINTS PARA GESTIÓN DE CONVERSACIONES
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
# ENDPOINTS ORIGINALES (MANTENIDOS)
# ===================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        service = await get_ultimate_cv_service()
        stats = await service.get_comprehensive_stats()
        return {
            "status": "healthy",
            "service_info": stats["service_info"],
            "components": {
                "connection_manager": "operational",
                "cache_system": "operational", 
                "chromadb_manager": "operational",
                "conversation_manager": "operational"
            },
            "features": {
                "conversational_memory": True,
                "multilingual_support": True,
                "automatic_language_detection": True
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics"""
    try:
        service = await get_ultimate_cv_service()
        return await service.get_comprehensive_stats()
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
