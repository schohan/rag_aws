# RAG Agent with AWS Bedrock

A production-ready AI Agent implementing Retrieval-Augmented Generation (RAG) using AWS Bedrock services, S3 Vector Storage, and DynamoDB. Built following Google ADK (Agent Development Kit) patterns for extensible, tool-based agent architecture.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Agent API                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   FastAPI    │  │     CLI      │  │   Streaming  │          │
│  │   Endpoints  │  │   Interface  │  │   Responses  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Agent Core                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Agent     │  │   Tools      │  │  Ingestion   │          │
│  │  Executor    │◄─┤   Registry   │  │   Pipeline   │          │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘          │
│         │              ▲                    │                   │
│         │    ┌─────────┴─────────┐          │                   │
│         │    │ • Vector Search   │          │                   │
│         │    │ • Document Tools  │          │                   │
│         │    │ • Web Search      │          │                   │
│         │    └───────────────────┘          │                   │
└─────────┼───────────────────────────────────┼───────────────────┘
          │                                   │
          ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AWS Services Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Bedrock    │  │  S3 Vectors  │  │   DynamoDB   │          │
│  │   (LLM +     │  │  (Vector     │  │  (Metadata   │          │
│  │  Embeddings) │  │   Storage)   │  │   Storage)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **🤖 Agentic RAG**: Tool-based agent following Google ADK patterns
- **🔍 Semantic Search**: S3 Vectors for efficient similarity search
- **📄 Document Ingestion**: Automatic chunking, embedding, and indexing
- **💬 Conversational**: Multi-turn conversations with memory
- **🔌 Extensible Tools**: Easy to add custom tools
- **🚀 Production Ready**: Async, scalable, with comprehensive error handling
- **☁️ AWS Native**: Leverages Bedrock, S3, and DynamoDB

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- AWS Account with Bedrock access
- AWS CLI configured

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag_aws.git
cd rag_aws

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp env.example .env
# Edit .env with your AWS credentials and settings
```

### Configuration

Edit `.env` with your settings:

```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# S3 Configuration
S3_BUCKET_NAME=your-bucket-name

# DynamoDB Configuration
DYNAMODB_TABLE_NAME=rag-agent-metadata

# Bedrock Model Configuration
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_LLM_MODEL_ID=qwen.qwen3-32b-v1:0
```

### Deploy Infrastructure

```bash
# Install CDK dependencies
pip install -e ".[deploy]"

# Deploy AWS infrastructure
cd infrastructure
cdk bootstrap  # First time only
cdk deploy --all
```
>Output is like this: <br>
API URL: https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/<br>
S3 Bucket: rag-agent-dev-documents-160755230655<br>
DynamoDB Table: rag-agent-dev-metadata


### Run the API

```bash
# Start the development server
rag-agent server --reload

# Or with specific host/port
rag-agent server --host 0.0.0.0 --port 8080
```

## 📖 Usage

### CLI Commands

```bash
# Set up infrastructure (onetime only.create buckets, etc)
rag-agent setup

# Start API server
rag-agent server

# Interactive chat
rag-agent chat

# Single query
rag-agent query "What is machine learning?"

# Ingest a document
rag-agent ingest document.txt --title "My Document"

# Ingest directory
rag-agent ingest-dir ./documents --pattern "*.md"

# list all documents
rag-agent query "list all documents"

```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | RAG query with sources |
| `/chat` | POST | Conversational chat |
| `/chat/stream` | POST | Streaming chat response |
| `/documents` | GET | List documents |
| `/documents` | POST | Ingest new document |
| `/documents/{id}` | GET | Get document details |
| `/documents/{id}` | DELETE | Delete document |
| `/tools` | GET | List available tools |

### Python SDK

```python
import asyncio
from rag_agent import RAGAgent
from rag_agent.models import QueryRequest

async def main():
    # Initialize agent
    agent = RAGAgent()
    
    # Simple chat
    response, conv_id = await agent.chat("What is RAG?")
    print(response)
    
    # Query with sources
    request = QueryRequest(
        query="Explain retrieval augmented generation",
        top_k=5,
        include_sources=True,
    )
    result = await agent.query(request)
    
    print(f"Answer: {result.answer}")
    for source in result.sources:
        print(f"  - {source.title} ({source.relevance_score:.0%})")

asyncio.run(main())
```

### Document Ingestion

```python
from rag_agent.ingestion import DocumentIngestionPipeline

async def ingest_documents():
    pipeline = DocumentIngestionPipeline()
    
    # Ingest text
    doc = await pipeline.ingest_text(
        content="Your document content here...",
        title="My Document",
        source="manual",
    )
    
    # Ingest file
    doc = await pipeline.ingest_file(
        file_path="./documents/guide.pdf",
        title="User Guide",
    )
    
    print(f"Document indexed: {doc.id}")
    print(f"Chunks created: {len(doc.chunks)}")
```

### Custom Tools

```python
from rag_agent.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

class CalculatorTool(Tool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            parameters=[
                ToolParameter(
                    name="expression",
                    description="Math expression to evaluate",
                    type="string",
                    required=True,
                ),
            ],
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        expr = kwargs.get("expression")
        try:
            result = eval(expr)  # Use safe eval in production!
            return ToolResult.success(data=result)
        except Exception as e:
            return ToolResult.error(str(e))

# Register with agent
agent = RAGAgent()
agent.register_tool(CalculatorTool())
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/rag_agent --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 📁 Project Structure

```
rag_aws/
├── src/rag_agent/
│   ├── __init__.py          # Package exports
│   ├── agent.py              # Main RAG agent
│   ├── api.py                # FastAPI application
│   ├── cli.py                # CLI commands
│   ├── config.py             # Settings management
│   ├── ingestion.py          # Document processing
│   ├── models.py             # Data models
│   ├── services/
│   │   ├── bedrock.py        # AWS Bedrock service
│   │   ├── dynamodb.py       # DynamoDB service
│   │   ├── embeddings.py     # Embedding generation
│   │   └── s3_vectors.py     # S3 vector storage
│   └── tools/
│       ├── base.py           # Tool base classes
│       ├── document.py       # Document tools
│       ├── search.py         # Search tools
│       └── web.py            # Web tools
├── infrastructure/
│   ├── app.py                # CDK app
│   ├── stack.py              # CDK stacks
│   └── cdk.json              # CDK config
├── tests/
│   ├── conftest.py           # Test fixtures
│   ├── test_models.py
│   ├── test_tools.py
│   └── test_ingestion.py
├── pyproject.toml            # Project config
├── env.example               # Environment template
└── README.md
```

## 🔧 Configuration Options

### Bedrock Models

| Model Type | Default | Alternatives |
|------------|---------|--------------|
| LLM | Claude 3 Sonnet | Claude 3 Haiku, Claude 3 Opus |
| Embeddings | Titan Embed v2 | Cohere Embed |

### Vector Settings

```python
# In your .env or config
VECTOR_DIMENSION=1024      # Match your embedding model
TOP_K_RESULTS=5            # Default search results
SIMILARITY_THRESHOLD=0.7   # Minimum similarity score
```

### Chunking Settings

```python
CHUNK_SIZE=1000           # Characters per chunk
CHUNK_OVERLAP=200         # Overlap between chunks
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [AWS Bedrock](https://aws.amazon.com/bedrock/) for foundation models
- [LangChain](https://langchain.com/) for inspiration
- Google ADK patterns for agent architecture

