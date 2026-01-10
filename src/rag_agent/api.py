"""
FastAPI REST API for the RAG Agent.

Provides HTTP endpoints for querying the agent, managing documents,
and handling conversations.
"""

from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_agent.agent import RAGAgent
from rag_agent.config import get_settings
from rag_agent.ingestion import DocumentIngestionPipeline
from rag_agent.models import QueryRequest, QueryResponse, DocumentStatus

logger = structlog.get_logger(__name__)


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., min_length=1, max_length=4096)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    conversation_id: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


class IngestTextRequest(BaseModel):
    """Text ingestion request model."""

    content: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=500)
    source: str = Field(default="manual_input")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Document response model."""

    id: str
    title: str
    source: str
    status: str
    chunk_count: int = 0
    created_at: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: dict[str, str]


# Global agent instance
_agent: RAGAgent | None = None
_pipeline: DocumentIngestionPipeline | None = None


def get_agent() -> RAGAgent:
    """Get the agent instance."""
    global _agent
    if _agent is None:
        _agent = RAGAgent()
    return _agent


def get_pipeline() -> DocumentIngestionPipeline:
    """Get the ingestion pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DocumentIngestionPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Agent API")
    global _agent, _pipeline
    _agent = RAGAgent()
    _pipeline = DocumentIngestionPipeline()
    
    # Create DynamoDB table if needed
    try:
        await _pipeline.dynamodb.create_table_if_not_exists()
    except Exception as e:
        logger.warning("Failed to create DynamoDB table", error=str(e))
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Agent API")


# Create FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="AI Agent with RAG capabilities using AWS Bedrock and S3 Vector Storage",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "bedrock": "configured" if settings.bedrock.llm_model_id else "not_configured",
            "dynamodb": "configured" if settings.dynamodb.table_name else "not_configured",
            "s3_vectors": "configured" if settings.s3.bucket_name else "not_configured",
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the RAG agent.
    
    Retrieves relevant context and generates an answer.
    """
    try:
        agent = get_agent()
        response = await agent.query(request)
        return response
    except Exception as e:
        logger.error("Query failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the RAG agent.
    
    Simpler conversational interface with optional conversation tracking.
    """
    try:
        agent = get_agent()
        conversation_id = request.conversation_id or str(uuid4())
        
        query_request = QueryRequest(
            query=request.message,
            include_sources=True,
        )
        
        response = await agent.query(query_request, conversation_id)
        
        return ChatResponse(
            response=response.answer,
            conversation_id=conversation_id,
            sources=[
                {
                    "title": s.title,
                    "source": s.source,
                    "snippet": s.snippet,
                    "relevance": s.relevance_score,
                }
                for s in response.sources
            ],
        )
    except Exception as e:
        logger.error("Chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def stream_chat_response(request: ChatRequest):
    """
    Stream a chat response.
    
    Returns a streaming response for real-time output.
    """
    try:
        agent = get_agent()
        
        async def generate():
            async for chunk in agent.stream_response(request.message):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.error("Stream chat failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/ingest", response_model=DocumentResponse)
async def ingest_document(
    request: IngestTextRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest a text document into the knowledge base.
    
    Processing happens in the background.
    """
    try:
        pipeline = get_pipeline()
        
        # Create document with pending status
        from rag_agent.models import Document
        document = Document(
            title=request.title,
            content=request.content,
            source=request.source,
            metadata=request.metadata,
        )
        
        # Process in background
        async def process():
            try:
                await pipeline.ingest_text(
                    content=request.content,
                    title=request.title,
                    source=request.source,
                    metadata=request.metadata,
                )
            except Exception as e:
                logger.error("Background ingestion failed", error=str(e))
        
        background_tasks.add_task(process)
        
        return DocumentResponse(
            id=str(document.id),
            title=document.title,
            source=document.source,
            status=DocumentStatus.PENDING.value,
            created_at=document.created_at.isoformat(),
        )
    except Exception as e:
        logger.error("Ingestion request failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload and ingest a file document.
    """
    try:
        import tempfile
        from pathlib import Path
        
        pipeline = get_pipeline()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            document = await pipeline.ingest_file(
                file_path=tmp_path,
                title=file.filename,
            )
            
            return DocumentResponse(
                id=str(document.id),
                title=document.title,
                source=document.source,
                status=document.status.value,
                chunk_count=len(document.chunks),
                created_at=document.created_at.isoformat(),
            )
        finally:
            tmp_path.unlink()  # Clean up temp file
            
    except Exception as e:
        logger.error("File upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentResponse])
async def list_documents(
    status: str | None = None,
    limit: int = 50,
):
    """
    List documents in the knowledge base.
    """
    try:
        pipeline = get_pipeline()
        doc_status = DocumentStatus(status) if status else DocumentStatus.INDEXED
        
        documents = await pipeline.dynamodb.list_documents_by_status(
            status=doc_status,
            limit=limit,
        )
        
        return [
            DocumentResponse(
                id=doc.id,
                title=doc.title,
                source=doc.source,
                status=doc.status.value,
                chunk_count=doc.chunk_count,
                created_at=doc.created_at,
            )
            for doc in documents
        ]
    except Exception as e:
        logger.error("List documents failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get a document by ID.
    """
    try:
        pipeline = get_pipeline()
        metadata = await pipeline.dynamodb.get_document_metadata(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": metadata.id,
            "title": metadata.title,
            "source": metadata.source,
            "status": metadata.status.value,
            "content_type": metadata.content_type,
            "chunk_count": metadata.chunk_count,
            "token_count": metadata.token_count,
            "created_at": metadata.created_at,
            "updated_at": metadata.updated_at,
            "indexed_at": metadata.indexed_at,
            "metadata": metadata.extra_metadata,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get document failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the knowledge base.
    """
    try:
        pipeline = get_pipeline()
        success = await pipeline.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"status": "deleted", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete document failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools")
async def list_tools():
    """
    List available agent tools.
    """
    agent = get_agent()
    tools = []
    
    for name in agent.list_tools():
        tool = agent.tools.get(name)
        if tool:
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "type": p.type,
                        "required": p.required,
                    }
                    for p in tool.definition.parameters
                ],
            })
    
    return {"tools": tools}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app

