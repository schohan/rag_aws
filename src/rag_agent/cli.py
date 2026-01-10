"""
Command Line Interface for the RAG Agent.

Provides CLI commands for running the API server, ingesting documents,
and interacting with the agent.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    import uvicorn
    from rag_agent.config import get_settings
    
    settings = get_settings()
    
    uvicorn.run(
        "rag_agent.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


async def ingest_file(file_path: str, title: Optional[str] = None):
    """Ingest a file into the knowledge base."""
    from rag_agent.ingestion import DocumentIngestionPipeline
    
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found", path=file_path)
        sys.exit(1)
    
    logger.info("Ingesting file", path=file_path)
    
    pipeline = DocumentIngestionPipeline()
    document = await pipeline.ingest_file(path, title=title)
    
    logger.info(
        "File ingested successfully",
        document_id=str(document.id),
        title=document.title,
        chunks=len(document.chunks),
        status=document.status.value,
    )
    
    return document


async def ingest_directory(directory: str, pattern: str = "*.txt"):
    """Ingest all matching files in a directory."""
    from rag_agent.ingestion import DocumentIngestionPipeline
    
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.error("Directory not found", path=directory)
        sys.exit(1)
    
    files = list(dir_path.glob(pattern))
    logger.info("Found files to ingest", count=len(files), pattern=pattern)
    
    pipeline = DocumentIngestionPipeline()
    
    for file_path in files:
        try:
            logger.info("Ingesting file", path=str(file_path))
            await pipeline.ingest_file(file_path)
        except Exception as e:
            logger.error("Failed to ingest file", path=str(file_path), error=str(e))
    
    logger.info("Directory ingestion complete", total_files=len(files))


async def chat_interactive():
    """Start an interactive chat session."""
    from rag_agent.agent import RAGAgent
    
    agent = RAGAgent()
    conversation_id = None
    
    print("\n🤖 RAG Agent Interactive Chat")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'new' to start a new conversation")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit"):
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == "new":
                conversation_id = None
                print("\n🔄 Starting new conversation...")
                continue
            
            # Get response from agent
            response, conversation_id = await agent.chat(user_input, conversation_id)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error("Chat error", error=str(e))
            print(f"\n❌ Error: {str(e)}")


async def query(question: str, top_k: int = 5):
    """Execute a single query."""
    from rag_agent.agent import RAGAgent
    from rag_agent.models import QueryRequest
    
    agent = RAGAgent()
    
    request = QueryRequest(
        query=question,
        top_k=top_k,
        include_sources=True,
    )
    
    response = await agent.query(request)
    
    print(f"\n📝 Question: {question}")
    print(f"\n🤖 Answer: {response.answer}")
    
    if response.sources:
        print(f"\n📚 Sources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            print(f"  {i}. {source.title} (score: {source.relevance_score:.2%})")
            print(f"     {source.snippet[:100]}...")
    
    print(f"\n⏱️ Latency: {response.latency_ms:.2f}ms")
    
    return response


async def setup_infrastructure():
    """Set up required AWS infrastructure."""
    from rag_agent.services.dynamodb import DynamoDBService
    from rag_agent.services.s3_vectors import S3VectorService
    
    logger.info("Setting up infrastructure...")
    
    # Create DynamoDB table
    dynamodb = DynamoDBService()
    try:
        created = await dynamodb.create_table_if_not_exists()
        if created:
            logger.info("DynamoDB table created")
        else:
            logger.info("DynamoDB table already exists")
    except Exception as e:
        logger.error("Failed to create DynamoDB table", error=str(e))
    
    # Create S3 vector index
    s3_vectors = S3VectorService()
    try:
        await s3_vectors.create_vector_index()
        logger.info("S3 vector index created")
    except Exception as e:
        logger.warning("Failed to create S3 vector index", error=str(e))
    
    logger.info("Infrastructure setup complete")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a single query")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of sources")
    
    # Ingest file command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file")
    ingest_parser.add_argument("file", help="File path to ingest")
    ingest_parser.add_argument("--title", help="Document title")
    
    # Ingest directory command
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="Ingest a directory")
    ingest_dir_parser.add_argument("directory", help="Directory path")
    ingest_dir_parser.add_argument("--pattern", default="*.txt", help="File pattern")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up AWS infrastructure")
    
    args = parser.parse_args()
    
    if args.command == "server":
        run_server(args.host, args.port, args.reload)
    
    elif args.command == "chat":
        asyncio.run(chat_interactive())
    
    elif args.command == "query":
        asyncio.run(query(args.question, args.top_k))
    
    elif args.command == "ingest":
        asyncio.run(ingest_file(args.file, args.title))
    
    elif args.command == "ingest-dir":
        asyncio.run(ingest_directory(args.directory, args.pattern))
    
    elif args.command == "setup":
        asyncio.run(setup_infrastructure())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

