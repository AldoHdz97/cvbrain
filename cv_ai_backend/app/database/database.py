"""
Database Configuration for Interview Scheduling
SQLite with SQLAlchemy 2.0 and async support
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

# Base class for all ORM models
Base = declarative_base()

class DatabaseManager:
    """
    Async SQLite Database Manager for Interview Scheduling
    
    Features:
    - Async SQLite support with aiosqlite
    - Automatic table creation
    - Session management
    - Connection pooling
    """
    
    def __init__(self, database_url: str = None):
        # Default to local SQLite file
        if database_url is None:
            db_path = Path("data/interviews.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/interviews.db")
        
        self.database_url = database_url
        self.engine = None
        self.async_session = None
        self._initialized = False
        
        logger.info(f"Database manager initialized with URL: {database_url}")
    
    async def initialize(self) -> bool:
        """Initialize async database engine and create tables"""
        try:
            logger.info("🗄️ Initializing SQLite database...")
            
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                poolclass=StaticPool,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 20
                }
            )
            
            # Create session factory
            self.async_session = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create all tables
            await self._create_tables()
            
            self._initialized = True
            logger.info("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
    
    async def _create_tables(self):
        """Create all database tables"""
        try:
            # Import models to register them with Base
            from app.models.interview_models import InterviewRequest
            
            async with self.engine.begin() as conn:
                # Create all tables defined in Base.metadata
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("📋 Database tables created/verified")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        if not self._initialized:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Check database health"""
        if not self._initialized:
            return False
        
        try:
            async with self.async_session() as session:
                # Simple query to test connection
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_stats(self) -> dict:
        """Get database statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            async with self.async_session() as session:
                # Import here to avoid circular imports
                from app.models.interview_models import InterviewRequest
                from sqlalchemy import func, select
                
                # Count total interviews
                total_result = await session.execute(
                    select(func.count(InterviewRequest.id))
                )
                total_interviews = total_result.scalar() or 0
                
                # Count pending interviews
                pending_result = await session.execute(
                    select(func.count(InterviewRequest.id)).where(
                        InterviewRequest.status == "pending"
                    )
                )
                pending_interviews = pending_result.scalar() or 0
                
                return {
                    "status": "healthy",
                    "total_interviews": total_interviews,
                    "pending_interviews": pending_interviews,
                    "database_url": self.database_url,
                    "engine_info": str(self.engine) if self.engine else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Clean up database connections"""
        logger.info("🧹 Cleaning up database connections...")
        
        if self.engine:
            await self.engine.dispose()
        
        self._initialized = False
        logger.info("✅ Database cleanup completed")

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get or create global database manager"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    db_manager = await get_database_manager()
    async for session in db_manager.get_session():
        yield session

async def cleanup_database():
    """Cleanup database connections on shutdown"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.cleanup()
        _db_manager = None
