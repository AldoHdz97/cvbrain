
"""
FastAPI Server for CV-AI Backend
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

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
        "message": "CV-AI Backend Ultimate v3.0", 
        "status": "operational",
        "docs": "/docs"
    }

@app.post("/query", response_model=UltimateQueryResponse)
async def query_cv(request: UltimateQueryRequest):
    """Query CV with AI-powered semantic search"""
    try:
        service = await get_ultimate_cv_service()
        response = await service.query_cv(request)
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                "chromadb_manager": "operational"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers
    )
