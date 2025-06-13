"""
CV Service - Core business logic
Simplified and clean implementation
"""

import asyncio
import logging
import time
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime

import chromadb
from openai import AsyncOpenAI

from app.core.config import settings
from app.models.schemas import QueryRequest, QueryResponse, QueryType

logger = logging.getLogger(__name__)

class CVService:
    """Core CV query service"""
    
    def __init__(self):
        self._initialized = False
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self._openai_client: Optional[AsyncOpenAI] = None
        self._cache: Dict[str, QueryResponse] = {}
        
    async def initialize(self) -> bool:
        """Initialize the service"""
        try:
            logger.info("Initializing CV Service...")
            
            # Initialize OpenAI client
            self._openai_client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_timeout
            )
            
            # Initialize ChromaDB
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            
            # Get or create collection
            try:
                self._collection = self._client.get_collection(
                    name=settings.chroma_collection_name
                )
            except Exception:
                logger.warning(f"Collection '{settings.chroma_collection_name}' not found")
                return False
            
            # Check if collection has data
            count = self._collection.count()
            if count == 0:
                logger.warning("Collection is empty - consider loading CV data")
                return False
            
            logger.info(f"CV Service initialized with {count} documents")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CV Service: {e}")
            return False
    
    async def query_cv(self, request: QueryRequest) -> QueryResponse:
        """Process CV query"""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(request.question)
            if settings.enable_caching and cache_key in self._cache:
                cached_response = self._cache[cache_key]
                cached_response.cache_hit = True
                return cached_response
            
            # Generate embedding
            embedding_response = await self._openai_client.embeddings.create(
                model=settings.embedding_model,
                input=request.question.strip()
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Search ChromaDB
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=request.k,
                include=['documents', 'distances', 'metadatas']
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            if not documents:
                return self._create_fallback_response(request, start_time)
            
            # Calculate similarity scores
            similarity_scores = [1.0 / (1.0 + dist) for dist in distances]
            
            # Generate AI response
            context = self._create_context(documents, similarity_scores)
            prompt = self._create_prompt(request.question, context, request.query_type)
            
            ai_response = await self._openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Aldo Hernández Villanueva, responding about your professional background in first person."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.openai_temperature,
                max_tokens=request.max_response_length
            )
            
            answer = ai_response.choices[0].message.content
            processing_time = time.time() - start_time
            
            # Create response
            response = QueryResponse(
                answer=answer,
                query_type=request.query_type or self._classify_query(request.question),
                relevant_chunks=len(documents),
                similarity_scores=similarity_scores,
                sources=documents[:3] if request.include_sources else [],
                processing_time=processing_time,
                model_used=settings.openai_model
            )
            
            # Cache response
            if settings.enable_caching:
                self._cache[cache_key] = response
                # Simple cache cleanup
                if len(self._cache) > 100:
                    # Remove oldest entries
                    keys_to_remove = list(self._cache.keys())[:20]
                    for key in keys_to_remove:
                        del self._cache[key]
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise
    
    def _get_cache_key(self, question: str) -> str:
        """Generate cache key for question"""
        return hashlib.md5(question.lower().encode()).hexdigest()
    
    def _classify_query(self, question: str) -> QueryType:
        """Simple query classification"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['skill', 'technology', 'programming', 'language', 'tool']):
            return QueryType.SKILLS
        elif any(word in question_lower for word in ['experience', 'work', 'job', 'company', 'role']):
            return QueryType.EXPERIENCE
        elif any(word in question_lower for word in ['education', 'degree', 'university', 'study']):
            return QueryType.EDUCATION
        elif any(word in question_lower for word in ['project', 'built', 'created', 'developed']):
            return QueryType.PROJECTS
        elif any(word in question_lower for word in ['contact', 'email', 'phone', 'linkedin']):
            return QueryType.CONTACT
        elif any(word in question_lower for word in ['summary', 'about', 'who', 'overview']):
            return QueryType.SUMMARY
        else:
            return QueryType.GENERAL
    
    def _create_context(self, documents: List[str], scores: List[float]) -> str:
        """Create context from documents"""
        context_parts = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Truncate long documents
            if len(doc) > 400:
                doc = doc[:400] + "..."
            context_parts.append(f"Context {i+1} (relevance: {score:.3f}):\n{doc}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str, query_type: Optional[QueryType]) -> str:
        """Create AI prompt"""
        base_prompt = """
Answer the following question about my professional background based on the provided context.
Respond in first person as Aldo Hernández Villanueva.
Be specific, professional, and engaging.
If the context doesn't contain enough information, say so politely.

Context:
{context}

Question: {question}

Answer:
"""
        return base_prompt.format(context=context, question=question)
    
    def _create_fallback_response(self, request: QueryRequest, start_time: float) -> QueryResponse:
        """Create fallback response when no relevant content found"""
        return QueryResponse(
            answer="I don't have enough relevant information to answer that specific question. Could you try asking about something else related to my professional background?",
            query_type=request.query_type or QueryType.GENERAL,
            relevant_chunks=0,
            similarity_scores=[],
            sources=[],
            processing_time=time.time() - start_time,
            model_used=settings.openai_model
        )
    
    async def cleanup(self):
        """Cleanup service resources"""
        if self._openai_client:
            await self._openai_client.close()
        self._initialized = False
        logger.info("CV Service cleanup completed")

# Service singleton
_service_instance: Optional[CVService] = None
_service_lock = asyncio.Lock()

async def get_cv_service() -> Optional[CVService]:
    """Get or create CV service instance"""
    global _service_instance
    
    async with _service_lock:
        if _service_instance is None:
            service = CVService()
            if await service.initialize():
                _service_instance = service
            else:
                await service.cleanup()
                return None
    
    return _service_instance
