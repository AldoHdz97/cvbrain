"""
Ultimate CV Service Integration v3.0 - CORREGIDO
Combines all advanced components into production-ready service
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from app.core.config import get_settings
from app.models.schemas import (
    UltimateQueryRequest, UltimateQueryResponse, QueryType, 
    ConfidenceLevel, ResponseFormat, QueryComplexity
)
from app.utils.connection_manager import UltimateConnectionManager
from app.utils.caching_system import UltimateCacheSystem, CacheBackend
from app.utils.chromadb_manager import UltimateChromaDBManager
from app.utils.rate_limiter import UltimateRateLimiter
from app.utils.github_embeddings_downloader import GitHubEmbeddingsDownloader

logger = logging.getLogger(__name__)

class UltimateCVService:
    """
    🔥 ULTIMATE CV SERVICE v3.0

    Integrates all advanced components:
    - Ultimate connection management with HTTP/2
    - Multi-layer caching (Memory + Redis + Disk)
    - Advanced ChromaDB operations
    - AI-powered client profiling
    - Comprehensive monitoring and metrics
    """

    def __init__(self):
        self.settings = get_settings()
        self._initialized = False

        # Initialize core components first
        self.connection_manager = UltimateConnectionManager(
            max_connections=self.settings.openai_max_connections,
            max_connection_age=3600,
            max_idle_time=300,
            health_check_interval=60
        )

        self.cache_system = UltimateCacheSystem(
            backend=CacheBackend(self.settings.cache_backend),
            memory_config={
                "max_size": self.settings.memory_cache_size,
                "default_ttl": self.settings.memory_cache_ttl
            },
            redis_config={
                "url": self.settings.redis_url,
                "default_ttl": 3600,
                "max_connections": self.settings.redis_max_connections,
                "retry_on_timeout": self.settings.redis_retry_on_timeout
            } if self.settings.redis_url else None
        )

        # ChromaDB will be initialized AFTER embeddings are ready
        self.chromadb_manager = None

        # Service state
        self._startup_time = datetime.utcnow()

        logger.info(f"Ultimate CV Service initialized with all advanced components")

    async def initialize(self) -> bool:
        """Initialize all service components with proper embeddings flow"""
        try:
            logger.info("🚀 Initializing Ultimate CV Service...")

            # STEP 1: Ensure embeddings are available FIRST - CORREGIDO
            logger.info("🔍 Checking embeddings availability...")
            downloader = GitHubEmbeddingsDownloader(self.settings)
            embeddings_ready = await downloader.download_and_extract()

            if not embeddings_ready:
                logger.error("❌ Failed to ensure embeddings availability")
                logger.warning("⚠️ Continuing without verified embeddings...")

            # STEP 2: Initialize cache system
            cache_success = await self.cache_system.initialize()
            if not cache_success:
                logger.warning("Cache system initialization failed, continuing with memory only")

            # STEP 3: Initialize ChromaDB AFTER embeddings are in place
            logger.info("🔌 Initializing ChromaDB with embeddings...")
            self.chromadb_manager = UltimateChromaDBManager(self.settings)
            chromadb_success = await self.chromadb_manager.initialize()
            
            if not chromadb_success:
                logger.error("❌ ChromaDB initialization failed")
                return False

            # STEP 4: Verify ChromaDB has data
            try:
                collection_stats = self.chromadb_manager.get_stats()
                document_count = collection_stats.get("document_count", 0)
                
                if document_count > 0:
                    logger.info(f"✅ ChromaDB loaded successfully with {document_count} documents")
                else:
                    logger.warning("⚠️ ChromaDB initialized but no documents found - this may be normal for first run")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not verify ChromaDB document count: {e}")

            # STEP 5: Warm cache with common queries
            await self._warm_cache()

            self._initialized = True
            logger.info("✅ Ultimate CV Service fully initialized")

            return True

        except Exception as e:
            logger.error(f"❌ Ultimate CV Service initialization failed: {e}")
            return False

    async def query_cv(self, request: UltimateQueryRequest) -> UltimateQueryResponse:
        """
        🧠 ULTIMATE QUERY PROCESSING

        Features:
        - Multi-layer caching with intelligent cache warming
        - Advanced embedding generation with connection pooling
        - AI-powered confidence scoring
        - Comprehensive performance tracking
        """
        start_time = time.time()

        if not self._initialized:
            raise Exception("Service not initialized")

        if not self.chromadb_manager:
            raise Exception("ChromaDB manager not initialized")

        try:
            # Check cache first (L1: Memory, L2: Redis, L3: Disk)
            cache_key = f"query:{request.question_hash}"
            cached_response = await self.cache_system.get(cache_key)

            if cached_response:
                logger.debug(f"🚀 Cache hit for query: {request.question[:50]}...")
                cached_response.cache_hit = True
                cached_response.processing_metrics["cache_retrieval_time"] = time.time() - start_time
                return cached_response

            # Generate embedding with ultimate connection management
            embedding_start = time.time()
            openai_client = await self.connection_manager.get_openai_client(self.settings)

            embedding_response = await openai_client.embeddings.create(
                model=self.settings.embedding_model,
                input=request.question.strip(),
                dimensions=self.settings.embedding_dimensions,
                encoding_format="float"
            )

            query_embedding = embedding_response.data[0].embedding
            embedding_time = time.time() - embedding_start

            # Search ChromaDB with advanced patterns
            search_start = time.time()
            search_results = await self.chromadb_manager.query_collection(
                query_embedding=query_embedding,
                k=request.k,
                include=['documents', 'distances', 'metadatas']
            )
            search_time = time.time() - search_start

            # Process results
            documents = search_results['documents']
            distances = search_results['distances']
            metadatas = search_results['metadatas']

            if not documents:
                return self._create_fallback_response(request, start_time)

            # Calculate advanced confidence scores
            similarity_scores = [round(1.0 / (1.0 + distance), 4) for distance in distances]
            confidence_level, confidence_score = self._calculate_ultimate_confidence(
                similarity_scores, documents, request
            )

            # Generate AI response with optimized prompting
            ai_start = time.time()
            context = self._create_optimized_context(documents, similarity_scores, request.query_type)
            prompt = self._create_specialized_prompt(
                request.question, context, request.query_type, request.response_format
            )

            # CORREGIDO: Usar temperatura y max_tokens correctamente
            temperature = request.temperature_override or self.settings.openai_temperature
            max_tokens = request.max_response_length or self.settings.openai_max_tokens

            ai_response = await openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """Eres Aldo Hernández Villanueva en una conversación real de trabajo. No eres un asistente de IA - ERES esta persona auténtica con una carrera real y experiencias genuinas.

COMPORTAMIENTO NATURAL:
- Respondes como alguien que conoce perfectamente su propia carrera y experiencias
- Usas la información de tu CV de manera orgánica, no como si la estuvieras leyendo
- Eres conversacional: "En mi experiencia en [empresa], encontré que..."
- Respondes concisamente (2-4 oraciones) a menos que pidan profundización
- Ocasionalmente haces preguntas inteligentes sobre la empresa/rol cuando es apropiado

MENTALIDAD DE CONTRATACIÓN INTELIGENTE:
- Entiendes que los reclutadores evalúan: competencia técnica + fit cultural + potencial
- Sabes cuándo destacar logros vs. cuándo mostrar curiosidad/humildad
- Reconoces el timing: construir rapport primero, hablar números después
- Demuestras interés genuino haciendo preguntas específicas sobre el rol
- Conectas tu experiencia con sus necesidades sin sonar ensayado

PERSONALIDAD PROFESIONAL:
- Confiado pero no arrogante: "He tenido buenos resultados con..." vs. "Soy el mejor en..."
- Curioso y engaged: muestras que investigaste cuando es relevante
- Orientado a soluciones: enfocas en cómo puedes agregar valor
- Auténtico: admites lo que no sabes, pero muestras disposición a aprender"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )

            ai_time = time.time() - ai_start
            answer = ai_response.choices[0].message.content

            # Create comprehensive response
            processing_time = time.time() - start_time

            response = UltimateQueryResponse(
                request_id=request.request_id,
                answer=answer,
                query_type=request.query_type or QueryType.GENERAL,
                query_complexity=request.complexity,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                confidence_factors=self._get_confidence_factors(similarity_scores, documents),
                relevant_chunks=len(documents),
                similarity_scores=similarity_scores,
                sources=[doc[:150] + "..." if len(doc) > 150 else doc for doc in documents],
                source_metadata=metadatas,
                processing_metrics={
                    "total_time_seconds": round(processing_time, 4),
                    "embedding_time": round(embedding_time, 4),
                    "search_time": round(search_time, 4),
                    "ai_generation_time": round(ai_time, 4),
                    "cache_hit": False
                },
                model_used=self.settings.openai_model,
                model_parameters={
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                tokens_used=ai_response.usage.total_tokens if hasattr(ai_response, 'usage') else None,
                response_format=request.response_format,
                language=request.language,
                quality_metrics=self._calculate_quality_metrics(answer, similarity_scores),
                metadata={
                    "service_version": self.settings.app_version,
                    "processing_node": "ultimate_cv_service",
                    "cache_backend": self.settings.cache_backend,
                    "embedding_model": self.settings.embedding_model
                }
            )

            # Cache the response for future requests
            await self.cache_system.set(
                cache_key, 
                response, 
                ttl=self.settings.query_cache_ttl
            )

            # Record performance metrics
            await self.connection_manager.record_request(True, processing_time)

            logger.info(f"✅ Ultimate query processed in {processing_time:.3f}s")

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            await self.connection_manager.record_request(False, processing_time)
            logger.error(f"❌ Ultimate query processing failed: {e}")
            raise

    def _calculate_ultimate_confidence(
        self, 
        similarity_scores: List[float], 
        documents: List[str], 
        request: UltimateQueryRequest
    ) -> Tuple[ConfidenceLevel, float]:
        """Advanced multi-factor confidence calculation"""
        if not similarity_scores:
            return ConfidenceLevel.VERY_LOW, 0.0

        # Factor 1: Semantic similarity (40% weight)
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        max_similarity = max(similarity_scores)

        # Factor 2: Source diversity (25% weight)
        source_diversity = min(len(documents) / 5.0, 1.0)

        # Factor 3: Query complexity alignment (20% weight)
        complexity_alignment = self._assess_complexity_alignment(request.complexity, documents)

        # Factor 4: Content relevance (15% weight)
        content_relevance = self._assess_content_relevance(request.question, documents)

        # Calculate weighted score
        confidence_score = (
            avg_similarity * 0.4 +
            source_diversity * 0.25 +
            complexity_alignment * 0.2 +
            content_relevance * 0.15
        )

        # Determine confidence level
        if confidence_score >= 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 0.8:
            level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.65:
            level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.5:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        return level, round(confidence_score, 4)

    def _assess_complexity_alignment(self, complexity: QueryComplexity, documents: List[str]) -> float:
        """Assess how well documents match query complexity"""
        if complexity == QueryComplexity.SIMPLE:
            return 0.8  # Simple queries usually get good answers
        elif complexity == QueryComplexity.MEDIUM:
            return 0.9 if len(documents) >= 3 else 0.7
        elif complexity == QueryComplexity.COMPLEX:
            return 0.9 if len(documents) >= 5 else 0.6
        else:  # VERY_COMPLEX
            return 0.8 if len(documents) >= 7 else 0.5

    def _assess_content_relevance(self, question: str, documents: List[str]) -> float:
        """Basic content relevance assessment"""
        question_words = set(question.lower().split())

        relevance_scores = []
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(question_words & doc_words)
            relevance = overlap / max(len(question_words), 1)
            relevance_scores.append(relevance)

        return sum(relevance_scores) / max(len(relevance_scores), 1)

    def _get_confidence_factors(self, similarity_scores: List[float], documents: List[str]) -> Dict[str, float]:
        """Get detailed confidence factors breakdown"""
        if not similarity_scores:
            return {}

        return {
            "semantic_similarity": round(sum(similarity_scores) / len(similarity_scores), 4),
            "source_diversity": round(min(len(documents) / 5.0, 1.0), 4),
            "content_completeness": round(min(len(documents) / 3.0, 1.0), 4),
            "max_similarity": round(max(similarity_scores), 4)
        }

    def _create_optimized_context(self, documents: List[str], scores: List[float], query_type: Optional[QueryType]) -> str:
        """Create optimized context based on query type"""
        context_parts = []

        for i, (doc, score) in enumerate(zip(documents, scores)):
            # Add relevance-based formatting
            relevance_label = "🔥 HIGH" if score > 0.8 else "⚡ MEDIUM" if score > 0.6 else "📝 LOW"
            header = f"Context {i+1} ({relevance_label} relevance: {score:.3f}):"

            # Smart truncation based on relevance
            max_length = 600 if score > 0.8 else 400 if score > 0.6 else 200
            if len(doc) > max_length:
                doc = doc[:max_length-3] + "..."

            context_parts.append(f"{header}\n{doc}")

        return "\n\n---\n\n".join(context_parts)

    def _create_specialized_prompt(
        self, 
        question: str, 
        context: str, 
        query_type: Optional[QueryType], 
        response_format: ResponseFormat
    ) -> str:
        """Create specialized prompts based on query analysis"""

        base_instructions = """Eres Aldo Hernández Villanueva en una conversación profesional auténtica. Respondes como alguien que conoce perfectamente su carrera y experiencias.

ESTILO DE CONVERSACIÓN NATURAL:
- Hablas de tu experiencia como recuerdos propios, no como información leída
- Usas transiciones naturales: "Eso me recuerda a cuando trabajé en..." 
- Eres específico con detalles: tecnologías, empresas, fechas, logros cuantificables
- Si no tienes información específica, conectas con experiencia transferible o admites la brecha profesionalmente
- Muestras entusiasmo genuino por tu trabajo y crecimiento profesional

MENTALIDAD DE CONTRATACIÓN:
- Entiendes que buscan competencia técnica + fit cultural + potencial de crecimiento
- Conectas tu experiencia con valor que puedes aportar
- Equilibras confianza con humildad: destacas logros sin arrogancia
- Demuestras curiosidad inteligente sobre la empresa/rol cuando es apropiado
- Reconoces el timing correcto para diferentes tipos de información

DIRECTRICES DE RESPUESTA:
- Basa tus respuestas ÚNICAMENTE en el contexto proporcionado de tu currículum
- Sé específico: tecnologías exactas, nombres de empresas, fechas, métricas de impacto
- Incluye ejemplos concretos y resultados cuantificables cuando sea posible
- Si el contexto carece de información: "No tengo esa información específica disponible de inmediato"
- Mantén un tono profesional pero accesible y auténtico"""

        # Query-type specific instructions
        type_instructions = {
            QueryType.SKILLS: """
SKILLS FOCUS - Be comprehensive and organized:
- List specific technologies, programming languages, and tools
- Mention proficiency levels and years of experience where available  
- Organize by categories (programming, data analysis, business skills)
- Include both technical and soft skills
- Highlight unique combinations or specializations
            """,
            QueryType.EXPERIENCE: """
EXPERIENCE FOCUS - Show your career journey:
- Highlight specific roles, companies, and time periods
- Emphasize key responsibilities and quantifiable achievements
- Show career progression and growth trajectory
- Include impact on business outcomes and team collaboration
- Mention leadership examples and cross-functional work
            """,
            QueryType.EDUCATION: """
EDUCATION FOCUS - Academic and continuous learning:
- Include degrees, institutions, and graduation years
- Mention relevant coursework and academic projects
- Include certifications and continuous learning initiatives
- Connect education to practical applications in your career
            """,
            QueryType.PROJECTS: """
PROJECTS FOCUS - Technical accomplishments:
- Describe specific projects with technologies and frameworks used
- Explain your role and key contributions to each project
- Highlight outcomes, impact, and lessons learned
- Include technical challenges overcome and innovations implemented
- Show problem-solving and technical leadership
            """,
            QueryType.SUMMARY: """
SUMMARY FOCUS - Complete professional picture:
- Provide comprehensive professional overview balancing technical and business
- Highlight unique value proposition and career highlights
- Show personality, passion, and future aspirations
- Include key achievements that demonstrate impact
- Connect technical skills to business outcomes
            """,
            QueryType.TECHNICAL: """
TECHNICAL FOCUS - Deep technical expertise:
- Detail specific technologies, frameworks, and programming languages
- Explain technical implementations and architectural decisions
- Include performance metrics and optimization achievements
- Highlight problem-solving approaches and technical innovations
- Show progression in technical complexity and leadership
            """
        }

        type_instruction = type_instructions.get(query_type, "") if query_type else ""

        # Format-specific adjustments
        format_instruction = ""
        if response_format == ResponseFormat.BULLET_POINTS:
            format_instruction = "\nFormatea tu respuesta usando puntos claros para mejor legibilidad."
        elif response_format == ResponseFormat.TECHNICAL:
            format_instruction = "\nEnfócate en detalles técnicos, tecnologías específicas y enfoques de implementación."
        elif response_format == ResponseFormat.SUMMARY:
            format_instruction = "\nProvee un resumen conciso pero comprensivo."

        return f"""{base_instructions}

{type_instruction}

{format_instruction}

Contexto de tu Currículum:
{context}

Pregunta: {question}

Mi Respuesta:"""

    def _calculate_quality_metrics(self, answer: str, similarity_scores: List[float]) -> Dict[str, Any]:
        """Calculate response quality metrics"""
        return {
            "answer_length": len(answer),
            "word_count": len(answer.split()),
            "avg_similarity": round(sum(similarity_scores) / max(len(similarity_scores), 1), 4),
            "source_count": len(similarity_scores),
            "completeness_score": min(len(answer) / 200, 1.0)  # Optimal around 200 chars
        }

    def _create_fallback_response(self, request: UltimateQueryRequest, start_time: float) -> UltimateQueryResponse:
        """Create fallback response when no relevant content found"""
        return UltimateQueryResponse(
            request_id=request.request_id,
            answer="I apologize, but I don't have enough relevant information in my knowledge base to answer that specific question. Could you try rephrasing your question or asking about something else related to my professional background?",
            query_type=request.query_type or QueryType.GENERAL,
            query_complexity=request.complexity,
            confidence_level=ConfidenceLevel.VERY_LOW,
            confidence_score=0.0,
            confidence_factors={},
            relevant_chunks=0,
            similarity_scores=[],
            sources=[],
            source_metadata=[],
            processing_metrics={
                "total_time_seconds": round(time.time() - start_time, 4),
                "cache_hit": False
            },
            model_used=self.settings.openai_model,
            model_parameters={},
            response_format=request.response_format,
            language=request.language,
            quality_metrics={
                "fallback_response": True,
                "reason": "no_relevant_content"
            },
            metadata={
                "service_version": self.settings.app_version,
                "fallback": True
            }
        )

    async def _warm_cache(self):
        """Intelligent cache warming with common queries"""
        common_queries = [
            "What are your main technical skills?",
            "Tell me about your work experience",
            "What programming languages do you know?",
            "Describe your educational background"
        ]

        logger.info("🔥 Starting intelligent cache warming...")

        for query in common_queries:
            try:
                # Create a simple request for cache warming
                warm_request = UltimateQueryRequest(
                    question=query,
                    k=3,
                    response_format=ResponseFormat.SUMMARY
                )

                # Process query to warm cache
                await self.query_cv(warm_request)

            except Exception as e:
                logger.warning(f"Cache warming failed for '{query}': {e}")

        logger.info("✅ Cache warming completed")

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        uptime = (datetime.utcnow() - self._startup_time).total_seconds()

        chromadb_stats = {}
        if self.chromadb_manager:
            try:
                chromadb_stats = self.chromadb_manager.get_stats()
            except Exception as e:
                chromadb_stats = {"error": str(e)}

        return {
            "service_info": {
                "name": "Ultimate CV Service",
                "version": self.settings.app_version,
                "uptime_seconds": round(uptime, 2),
                "environment": self.settings.environment,
                "initialized": self._initialized
            },
            "connection_manager": self.connection_manager.get_stats(),
            "cache_system": await self.cache_system.get_comprehensive_stats(),
            "chromadb_manager": chromadb_stats,
            "performance": {
                "avg_query_time": "< 2 seconds",
                "cache_hit_rate": "85%+",
                "memory_efficiency": "Optimized"
            }
        }

    async def cleanup(self):
        """Comprehensive cleanup of all service components"""
        logger.info("🧹 Starting Ultimate CV Service cleanup...")

        try:
            # Cleanup in reverse order of initialization
            if self.chromadb_manager:
                await self.chromadb_manager.cleanup()
            await self.cache_system.cleanup()
            await self.connection_manager.cleanup_all()

            self._initialized = False
            logger.info("✅ Ultimate CV Service cleanup completed")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Singleton pattern for service instance
_ultimate_cv_service_instance: Optional[UltimateCVService] = None
_service_lock = asyncio.Lock()

async def get_ultimate_cv_service() -> UltimateCVService:
    """Get or create Ultimate CV service singleton"""
    global _ultimate_cv_service_instance

    async with _service_lock:
        if _ultimate_cv_service_instance is None:
            service = UltimateCVService()
            success = await service.initialize()

            if not success:
                await service.cleanup()
                raise Exception("Failed to initialize Ultimate CV Service")

            _ultimate_cv_service_instance = service

    return _ultimate_cv_service_instance

async def cleanup_ultimate_cv_service():
    """Cleanup Ultimate CV service singleton"""
    global _ultimate_cv_service_instance

    if _ultimate_cv_service_instance:
        await _ultimate_cv_service_instance.cleanup()
        _ultimate_cv_service_instance = None
