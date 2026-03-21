"""
FastAPI application for Financial Intelligence RAG system.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid
from loguru import logger
from pathlib import Path
from typing import Optional
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.models import (
    QueryRequest,
    QueryResponse,
    Citation,
    HealthResponse,
    ErrorResponse,
    DocumentUploadResponse
)
from src.config import get_settings, RAW_DATA_DIR
from src.ingestion.embedding_service import EmbeddingService
from src.retrieval.vector_store import QdrantVectorStore
from src.retrieval.bm25_search import BM25KeywordSearch
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.citation_engine import CitationEngine
from src.agents.rag_orchestrator import RAGOrchestrator
from src.agents.query_expander import QueryExpander

# Initialize settings
settings = get_settings()

# Configure logging
logger.add(
    "logs/api.log",
    rotation="500 MB",
    retention="10 days",
    level=settings.log_level
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Financial Intelligence RAG API for analyzing SEC 10-K filings"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for services
class AppState:
    """Application state container."""

    embedding_service: Optional[EmbeddingService] = None
    vector_store: Optional[QdrantVectorStore] = None
    bm25_search: Optional[BM25KeywordSearch] = None
    hybrid_retriever: Optional[HybridRetriever] = None
    citation_engine: Optional[CitationEngine] = None
    rag_orchestrator: Optional[RAGOrchestrator] = None
    query_expander: Optional[QueryExpander] = None
    initialized: bool = False


state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Financial Intelligence RAG API...")

    try:
        # Initialize embedding service
        state.embedding_service = EmbeddingService(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            cache_dir=Path("data/embeddings")
        )
        logger.info("Embedding service initialized")

        # Initialize vector store
        state.vector_store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            vector_size=state.embedding_service.get_embedding_dimension()
        )
        logger.info("Vector store initialized")

        # Initialize BM25 search
        state.bm25_search = BM25KeywordSearch()
        logger.info("BM25 search initialized")

        # Initialize hybrid retriever
        state.hybrid_retriever = HybridRetriever(
            vector_store=state.vector_store,
            bm25_search=state.bm25_search,
            embedding_service=state.embedding_service,
            rrf_k=settings.rrf_k
        )
        logger.info("Hybrid retriever initialized")

        # Initialize citation engine
        state.citation_engine = CitationEngine()
        logger.info("Citation engine initialized")

        # Initialize RAG orchestrator
        llm_model = settings.anthropic_model if settings.llm_provider == "anthropic" else settings.openai_model
        state.rag_orchestrator = RAGOrchestrator(
            hybrid_retriever=state.hybrid_retriever,
            citation_engine=state.citation_engine,
            llm_model=llm_model,
            llm_provider=settings.llm_provider,
            temperature=0.0
        )
        logger.info(f"RAG orchestrator initialized with {settings.llm_provider} ({llm_model})")

        # Initialize query expander
        state.query_expander = QueryExpander(
            llm_model=llm_model,
            llm_provider=settings.llm_provider
        )
        logger.info(f"Query expander initialized with {settings.llm_provider} ({llm_model})")

        state.initialized = True
        logger.info("✓ All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        state.initialized = False


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Intelligence RAG API",
        "version": settings.app_version,
        "status": "running" if state.initialized else "initializing"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    qdrant_connected = False
    embeddings_available = False

    try:
        if state.vector_store:
            info = state.vector_store.get_collection_info()
            qdrant_connected = info.get("status") is not None

        if state.embedding_service:
            embeddings_available = True

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")

    return HealthResponse(
        status="healthy" if state.initialized else "initializing",
        version=settings.app_version,
        qdrant_connected=qdrant_connected,
        embeddings_available=embeddings_available
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint for asking questions about the 10-K filing.
    Now uses RAG Orchestrator with LLM for answer generation.
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Services not initialized")

    start_time = time.time()
    query_id = str(uuid.uuid4())

    try:
        logger.info(f"Processing query [{query_id}]: {request.question}")

        # Check if query expansion is needed
        if request.expand_query and state.query_expander:
            query_analysis = state.query_expander.process_query(
                query=request.question,
                auto_expand=True
            )

            # If vague and expanded, handle multi-query
            if query_analysis.get("is_vague") and len(query_analysis.get("sub_questions", [])) > 1:
                logger.info(f"Expanding vague query into {len(query_analysis['sub_questions'])} sub-questions")

                result = state.rag_orchestrator.multi_query_answer(
                    questions=query_analysis["sub_questions"],
                    top_k_per_query=3
                )

                answer = result["answer"]
                calculation_steps = None
                citations = []
                confidence = 0.8

            else:
                # Use regular single-query flow
                result = state.rag_orchestrator.answer_question(
                    question=request.question,
                    top_k_final=request.top_k
                )

                answer = result["answer"]
                calculation_steps = result.get("calculation_steps")
                citations = result.get("citations", [])
                confidence = result.get("confidence", 0.0)

        else:
            # Regular query without expansion
            result = state.rag_orchestrator.answer_question(
                question=request.question,
                top_k_final=request.top_k
            )

            answer = result["answer"]
            calculation_steps = result.get("calculation_steps")
            citations = result.get("citations", [])
            confidence = result.get("confidence", 0.0)

        # Convert citations to Citation models
        citation_models = []
        for c in citations[:request.max_sources]:
            if isinstance(c, dict):
                citation_models.append(
                    Citation(
                        citation_id=c.get("citation_id", str(uuid.uuid4())),
                        text=c.get("text", ""),
                        page_number=c.get("page_number", 0),
                        section=c.get("section", "Unknown"),
                        score=c.get("relevance_score", 0.0),
                        chunk_type=c.get("chunk_type", "text")
                    )
                )

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)

        # Determine status
        if "NOT_FOUND" in answer:
            status = "no_results"
        elif "ERROR" in answer:
            status = "error"
        else:
            status = "success"

        return QueryResponse(
            query_id=query_id,
            question=request.question,
            answer=answer,
            calculation_steps=calculation_steps,
            citations=citation_models,
            confidence=confidence,
            processing_time_ms=processing_time,
            status=status
        )

    except Exception as e:
        logger.error(f"Error processing query [{query_id}]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a 10-K PDF document for ingestion.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save the uploaded file
        file_path = RAW_DATA_DIR / file.filename
        content = await file.read()

        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(f"Uploaded document: {file.filename}")

        return DocumentUploadResponse(
            document_id=str(uuid.uuid4()),
            filename=file.filename,
            status="uploaded",
            message=f"Document uploaded successfully. Use /api/documents/ingest to process it."
        )

    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collection/info")
async def collection_info():
    """Get information about the vector store collection."""
    if not state.vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        info = state.vector_store.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
