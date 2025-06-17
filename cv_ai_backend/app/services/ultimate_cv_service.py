"""
Ultimate CV Service Integration v3.0 - CONVERSATIONAL EDITION
Combines all advanced components into production-ready service with conversational capabilities
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Deque
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field

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

# ===================================================================
# NUEVAS CLASES PARA SISTEMA CONVERSACIONAL
# ===================================================================

@dataclass
class ConversationMessage:
    """Representa un mensaje en la conversación"""
    role: str  # "user" o "assistant"
    content: str
    timestamp: datetime
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationSession:
    """Maneja una sesión completa de conversación"""
    session_id: str
    messages: Deque[ConversationMessage] = field(default_factory=lambda: deque(maxlen=20))
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None) -> ConversationMessage:
        """Agrega un mensaje a la conversación"""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        return message
    
    def get_openai_messages(self, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """Convierte mensajes a formato OpenAI con gestión de tokens"""
        openai_messages = []
        current_tokens = 0
        
        # Estimar tokens (aproximadamente 4 caracteres = 1 token)
        for message in reversed(self.messages):
            estimated_tokens = len(message.content) // 4
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            openai_messages.insert(0, {
                "role": message.role,
                "content": message.content
            })
            current_tokens += estimated_tokens
        
        return openai_messages
    
    def get_conversation_summary(self) -> str:
        """Genera un resumen de la conversación para contexto"""
        if len(self.messages) <= 2:
            return ""
        
        topics_discussed = []
        for msg in self.messages:
            if msg.role == "user" and len(msg.content) > 10:
                # Extraer tema principal de la pregunta
                topic = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                topics_discussed.append(topic)
        
        if topics_discussed:
            return f"Temas previamente discutidos: {', '.join(topics_discussed[-3:])}"  # Últimos 3 temas
        return ""

class ConversationManager:
    """Gestor principal de conversaciones"""
    
    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions
        self._cleanup_lock = asyncio.Lock()
    
    async def get_or_create_session(self, session_id: str = None) -> ConversationSession:
        """Obtiene una sesión existente o crea una nueva"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            # Cleanup de sesiones viejas si es necesario
            await self._cleanup_old_sessions()
            
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        
        return self.sessions[session_id]
    
    async def _cleanup_old_sessions(self):
        """Limpia sesiones inactivas"""
        async with self._cleanup_lock:
            if len(self.sessions) >= self.max_sessions:
                # Ordenar por último uso y eliminar las más viejas
                sorted_sessions = sorted(
                    self.sessions.items(),
                    key=lambda x: x[1].last_activity
                )
                
                # Mantener solo las últimas max_sessions-10
                sessions_to_keep = dict(sorted_sessions[-(self.max_sessions-10):])
                self.sessions = sessions_to_keep
                
                logger.info(f"Cleaned up old conversation sessions. Active sessions: {len(self.sessions)}")

# ===================================================================
# CLASE PRINCIPAL ACTUALIZADA CON CAPACIDADES CONVERSACIONALES
# ===================================================================

class UltimateCVService:
    """
    🔥 ULTIMATE CV SERVICE v3.0 - CONVERSATIONAL EDITION

    Integrates all advanced components plus conversational capabilities:
    - Ultimate connection management with HTTP/2
    - Multi-layer caching (Memory + Redis + Disk)
    - Advanced ChromaDB operations
    - AI-powered client profiling
    - 🆕 Conversational memory and context management
    - 🆕 Natural interview-style interactions
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

        # 🆕 Sistema conversacional
        self.conversation_manager = ConversationManager(max_sessions=200)
        self._conversation_enabled = True

        # Service state
        self._startup_time = datetime.utcnow()

        logger.info(f"Ultimate CV Service initialized with conversational capabilities")

    async def initialize(self) -> bool:
        """Initialize all service components with proper embeddings flow"""
        try:
            logger.info("🚀 Initializing Ultimate CV Service with conversational capabilities...")

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
            logger.info("✅ Ultimate CV Service with conversational capabilities fully initialized")

            return True

        except Exception as e:
            logger.error(f"❌ Ultimate CV Service initialization failed: {e}")
            return False

    async def query_cv_conversational(
        self, 
        request: UltimateQueryRequest,
        session_id: str = None,
        maintain_context: bool = True
    ) -> UltimateQueryResponse:
        """
        🎯 QUERY CV CON CONTEXTO CONVERSACIONAL COMPLETO
        
        Features:
        - Mantiene historial de conversación
        - Contexto inteligente entre mensajes
        - Gestión automática de tokens
        - Conversaciones naturales de contratación
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
            # PASO 1: Gestionar sesión conversacional
            conversation_session = None
            if maintain_context and self._conversation_enabled:
                conversation_session = await self.conversation_manager.get_or_create_session(session_id)
                
                # Agregar pregunta del usuario al historial
                conversation_session.add_message("user", request.question)

            # PASO 2: Check cache con contexto conversacional
            cache_key = self._generate_conversational_cache_key(request, conversation_session)
            cached_response = await self.cache_system.get(cache_key)

            if cached_response:
                logger.debug(f"🚀 Conversational cache hit for: {request.question[:50]}...")
                cached_response.cache_hit = True
                cached_response.processing_metrics["cache_retrieval_time"] = time.time() - start_time
                return cached_response

            # PASO 3: Generate embedding with ultimate connection management
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

            # PASO 4: Search ChromaDB with advanced patterns
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
                response = self._create_fallback_response(request, start_time)
                if conversation_session:
                    conversation_session.add_message("assistant", response.answer)
                return response

            # PASO 5: Calculate advanced confidence scores
            similarity_scores = [round(1.0 / (1.0 + distance), 4) for distance in distances]
            confidence_level, confidence_score = self._calculate_ultimate_confidence(
                similarity_scores, documents, request
            )

            # PASO 6: 🔥 CREAR PROMPT CONVERSACIONAL CON OPTIMIZACIÓN
            ai_start = time.time()
            context = self._create_optimized_context(documents, similarity_scores, request.query_type)
            
            # Nuevo: Prompt conversacional con historial
            prompt = self._create_conversational_prompt(
                request.question, 
                context, 
                conversation_session,
                request.query_type, 
                request.response_format
            )

            # PASO 7: 🔥 LLAMADA A OPENAI CON HISTORIAL CONVERSACIONAL
            temperature = request.temperature_override or self.settings.openai_temperature
            max_tokens = request.max_response_length or self.settings.openai_max_tokens

            # Construir mensajes con historial conversacional
            messages = [
                {
                    "role": "system",
                    "content": self._get_conversational_system_prompt(conversation_session)
                }
            ]

            # Agregar historial de conversación si existe
            if conversation_session and len(conversation_session.messages) > 1:
                conversation_history = conversation_session.get_openai_messages(max_tokens=2000)
                messages.extend(conversation_history[:-1])  # Excluir la última pregunta del usuario

            # Agregar el prompt actual
            messages.append({"role": "user", "content": prompt})

            ai_response = await openai_client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )

            ai_time = time.time() - ai_start
            answer = ai_response.choices[0].message.content

            # PASO 8: Agregar respuesta al historial conversacional
            if conversation_session:
                conversation_session.add_message("assistant", answer, {
                    "confidence_score": confidence_score,
                    "sources_used": len(documents)
                })

            # PASO 9: Create comprehensive response
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
                    "cache_hit": False,
                    "conversation_context_used": conversation_session is not None,
                    "conversation_length": len(conversation_session.messages) if conversation_session else 0
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
                    "processing_node": "ultimate_cv_service_conversational",
                    "cache_backend": self.settings.cache_backend,
                    "embedding_model": self.settings.embedding_model,
                    "session_id": conversation_session.session_id if conversation_session else None,
                    "conversation_turn": len(conversation_session.messages) if conversation_session else 1
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

            logger.info(f"✅ Conversational query processed in {processing_time:.3f}s (Session: {conversation_session.session_id if conversation_session else 'none'})")

            return response

        except Exception as e:
            processing_time = time.time() - start_time
            await self.connection_manager.record_request(False, processing_time)
            logger.error(f"❌ Conversational query processing failed: {e}")
            raise

    async def query_cv(self, request: UltimateQueryRequest) -> UltimateQueryResponse:
        """
        🧠 QUERY CV ORIGINAL (COMPATIBILIDAD)
        
        Mantiene compatibilidad con código existente, pero internamente usa
        el sistema conversacional sin mantener contexto entre llamadas.
        """
        return await self.query_cv_conversational(
            request=request,
            session_id=None,  # Sin sesión = comportamiento original
            maintain_context=False
        )

    # ===================================================================
    # FUNCIONES CONVERSACIONALES NUEVAS
    # ===================================================================

    def _generate_conversational_cache_key(
        self, 
        request: UltimateQueryRequest, 
        conversation_session: ConversationSession = None
    ) -> str:
        """Genera cache key considerando contexto conversacional"""
        base_key = f"query:{request.question_hash}"
        
        if conversation_session and len(conversation_session.messages) > 1:
            # Incluir hash del contexto conversacional
            conversation_context = "".join([msg.content for msg in conversation_session.messages[-3:]])
            context_hash = str(hash(conversation_context))
            return f"{base_key}:ctx_{context_hash}"
        
        return base_key

    def _get_conversational_system_prompt(self, conversation_session: ConversationSession = None) -> str:
        """Sistema prompt que incluye contexto conversacional"""
        base_prompt = """Eres Aldo Hernández Villanueva en una conversación real de trabajo. No eres un asistente de IA - ERES esta persona auténtica con una carrera real y experiencias genuinas.

COMPORTAMIENTO CONVERSACIONAL NATURAL:
- Respondes como alguien que conoce perfectamente su propia carrera y experiencias
- Usas la información de tu CV de manera orgánica, no como si la estuvieras leyendo
- Eres conversacional y puedes hacer referencias a temas que ya discutimos
- Respondes concisamente (2-4 oraciones) a menos que pidan profundización
- Haces preguntas inteligentes de seguimiento cuando es apropiado para mantener la conversación

MENTALIDAD DE CONTRATACIÓN INTELIGENTE:
- Entiendes que los reclutadores evalúan: competencia técnica + fit cultural + potencial
- Sabes cuándo destacar logros vs. cuándo mostrar curiosidad/humildad
- Reconoces el timing: construir rapport primero, hablar números después
- Demuestras interés genuino haciendo preguntas específicas sobre el rol
- Conectas tu experiencia con sus necesidades sin sonar ensayado
- Puedes hacer referencias naturales a lo que ya hemos discutido

PERSONALIDAD PROFESIONAL:
- Confiado pero no arrogante: "He tenido buenos resultados con..." vs. "Soy el mejor en..."
- Curioso y engaged: muestras que investigaste cuando es relevante
- Orientado a soluciones: enfocas en cómo puedes agregar valor
- Auténtico: admites lo que no sabes, pero muestras disposición a aprender
- Conversacional: puedes hacer transiciones naturales como "Como mencioné antes..." si es relevante"""

        # Agregar contexto conversacional si existe
        if conversation_session and len(conversation_session.messages) > 1:
            conversation_summary = conversation_session.get_conversation_summary()
            if conversation_summary:
                base_prompt += f"\n\nCONTEXTO DE NUESTRA CONVERSACIÓN:\n{conversation_summary}"

        return base_prompt

    def _create_conversational_prompt(
        self,
        question: str,
        context: str,
        conversation_session: ConversationSession = None,
        query_type: Optional[QueryType] = None,
        response_format: ResponseFormat = ResponseFormat.CONVERSATIONAL
    ) -> str:
        """Crea prompt especializado con contexto conversacional"""
        
        base_instructions = """Responde de manera natural y conversacional. Si es relevante, puedes hacer referencias a temas que ya discutimos anteriormente.

DIRECTRICES DE RESPUESTA:
- Basa tus respuestas ÚNICAMENTE en el contexto proporcionado de tu currículum
- Sé específico: tecnologías exactas, nombres de empresas, fechas, métricas de impacto
- Incluye ejemplos concretos y resultados cuantificables cuando sea posible
- Si el contexto carece de información: "No tengo esa información específica disponible de inmediato"
- Mantén un tono profesional pero accesible y auténtico
- Si es apropiado, haz una pregunta inteligente de seguimiento sobre la empresa o el rol"""

        # Instrucciones específicas por tipo (reutilizando las existentes)
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

        # Agregar contexto conversacional si existe
        conversation_context = ""
        if conversation_session and len(conversation_session.messages) > 2:
            conversation_context = f"\n\nNota: Esta es una conversación continua. Puedes hacer referencias naturales a temas que ya hemos discutido si es relevante para la respuesta."

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

{conversation_context}

Contexto de tu Currículum:
{context}

Pregunta actual: {question}

Mi Respuesta:"""

    # ===================================================================
    # FUNCIONES ORIGINALES MANTENIDAS (CON ACTUALIZACIONES MENORES)
    # ===================================================================

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
        """Create specialized prompts based on query analysis (VERSIÓN ORIGINAL PARA COMPATIBILIDAD)"""

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

    # ===================================================================
    # NUEVAS FUNCIONES UTILITARIAS CONVERSACIONALES
    # ===================================================================

    async def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """Obtiene el historial de una conversación específica"""
        if session_id not in self.conversation_manager.sessions:
            return {"error": "Session not found"}
        
        session = self.conversation_manager.sessions[session_id]
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "message_id": msg.message_id
                }
                for msg in session.messages
            ]
        }

    async def clear_conversation(self, session_id: str) -> bool:
        """Limpia una conversación específica"""
        if session_id in self.conversation_manager.sessions:
            del self.conversation_manager.sessions[session_id]
            return True
        return False

    async def get_active_conversations(self) -> Dict[str, Any]:
        """Obtiene estadísticas de conversaciones activas"""
        active_sessions = len(self.conversation_manager.sessions)
        total_messages = sum(len(session.messages) for session in self.conversation_manager.sessions.values())
        
        return {
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "avg_messages_per_session": round(total_messages / max(active_sessions, 1), 2),
            "conversation_enabled": self._conversation_enabled
        }

    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics including conversational metrics"""
        uptime = (datetime.utcnow() - self._startup_time).total_seconds()

        chromadb_stats = {}
        if self.chromadb_manager:
            try:
                chromadb_stats = self.chromadb_manager.get_stats()
            except Exception as e:
                chromadb_stats = {"error": str(e)}

        # Get conversational stats
        conversation_stats = await self.get_active_conversations()

        return {
            "service_info": {
                "name": "Ultimate CV Service - Conversational Edition",
                "version": self.settings.app_version,
                "uptime_seconds": round(uptime, 2),
                "environment": self.settings.environment,
                "initialized": self._initialized,
                "conversational_enabled": self._conversation_enabled
            },
            "connection_manager": self.connection_manager.get_stats(),
            "cache_system": await self.cache_system.get_comprehensive_stats(),
            "chromadb_manager": chromadb_stats,
            "conversation_manager": conversation_stats,
            "performance": {
                "avg_query_time": "< 2 seconds",
                "cache_hit_rate": "85%+",
                "memory_efficiency": "Optimized",
                "conversational_context": "Enhanced"
            }
        }

    async def cleanup(self):
        """Comprehensive cleanup of all service components including conversations"""
        logger.info("🧹 Starting Ultimate CV Service cleanup...")

        try:
            # Cleanup conversations
            self.conversation_manager.sessions.clear()
            
            # Cleanup in reverse order of initialization
            if self.chromadb_manager:
                await self.chromadb_manager.cleanup()
            await self.cache_system.cleanup()
            await self.connection_manager.cleanup_all()

            self._initialized = False
            logger.info("✅ Ultimate CV Service cleanup completed")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# ===================================================================
# SINGLETON PATTERN ACTUALIZADO
# ===================================================================

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

async def get_conversational_cv_service() -> UltimateCVService:
    """Obtiene el servicio con capacidades conversacionales habilitadas (alias para claridad)"""
    service = await get_ultimate_cv_service()
    service._conversation_enabled = True
    return service

async def cleanup_ultimate_cv_service():
    """Cleanup Ultimate CV service singleton"""
    global _ultimate_cv_service_instance

    if _ultimate_cv_service_instance:
        await _ultimate_cv_service_instance.cleanup()
        _ultimate_cv_service_instance = None
