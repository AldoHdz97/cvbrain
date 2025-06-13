"""
CV-AI Backend Main Application
Clean and production-ready FastAPI application
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.models.schemas import QueryRequest, QueryResponse, HealthResponse
from app.services.cv_service import CVService, get_cv_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting CV-AI Backend...")
    
    # Initialize services
    service = await get_cv_service()
    if not service:
        logger.error("❌ Failed to initialize CV service")
        raise RuntimeError("Service initialization failed")
    
    logger.info("✅ CV-AI Backend ready")
    
    try:
        yield
    finally:
        logger.info("🛑 Shutting down CV-AI Backend...")
        if service:
            await service.cleanup()
        logger.info("✅ Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="AI-powered CV query system",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        **settings.get_cors_config()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"{settings.app_name} v{settings.app_version}",
        "status": "operational",
        "docs": "/docs"
    }

@app.post("/query", response_model=QueryResponse)
async def query_cv(request: QueryRequest):
    """Query CV with AI-powered search"""
    try:
        service = await get_cv_service()
        if not service:
            raise HTTPException(status_code=503, detail="Service unavailable")
        
        response = await service.query_cv(request)
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query processing failed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        service = await get_cv_service()
        
        components = {}
        if service:
            components["cv_service"] = "operational"
            components["chromadb"] = "operational"
        else:
            components["cv_service"] = "unavailable"
        
        return HealthResponse(
            status="healthy" if service else "unhealthy",
            version=settings.app_version,
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            components={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
