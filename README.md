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
- **🛠️ Infrastructure Management**: Built-in utilities to list and manage CDK bootstrap resources

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
cdk bootstrap  # First time only - creates CDKToolkit stack
cdk deploy --all
```

**Output example:**
```
API URL: https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/
S3 Bucket: rag-agent-dev-documents-160755230655
DynamoDB Table: rag-agent-dev-metadata
```

**What gets created:**
- **Bootstrap Stack** (`CDKToolkit`): CDK deployment infrastructure (one-time setup)
- **Application Stacks**:
  - `RAGAgentStack-dev`: S3 buckets, DynamoDB tables, IAM roles
  - `RAGAgentLambdaStack-dev`: API Gateway, Lambda functions

### Teardown Infrastructure

The `cdk-teardown` command is the opposite of `cdk bootstrap` and `cdk deploy` - it removes all infrastructure in a safe, automated way.

#### Quick Start

```bash
cd infrastructure

# Teardown all application stacks (uses standard 'cdk destroy')
./cdk-teardown --all

# Teardown specific environment
./cdk-teardown --environment dev

# Teardown application stacks AND bootstrap stack
./cdk-teardown --all --bootstrap

# Teardown with automatic resource cleanup (empties S3/ECR)
./cdk-teardown --all --cleanup

# List what would be removed (dry run)
./cdk-teardown --list
```

#### Understanding Bootstrap vs Application Stacks

**Bootstrap Stack (`CDKToolkit`)**:
- Created by `cdk bootstrap`
- Contains CDK deployment infrastructure (S3 for assets, ECR for images, IAM roles)
- Required for CDK deployments to work
- Shared across all CDK projects in an account/region

**Application Stacks** (`RAGAgentStack-*`, `RAGAgentLambdaStack-*`):
- Created by `cdk deploy`
- Contains your actual application resources:
  - API Gateway (`rag-agent-dev-api`)
  - Lambda functions
  - S3 buckets for documents
  - DynamoDB tables
  - IAM roles for the application
- Environment-specific (dev, prod, etc.)

#### Teardown Options

| Option | Description |
|--------|-------------|
| `--all` | Remove all application stacks (uses `cdk destroy --all`) |
| `--environment ENV` | Remove stacks for specific environment (e.g., `dev`, `prod`) |
| `--bootstrap` | Also remove CDK bootstrap stack (CDKToolkit) |
| `--force` | Skip confirmation prompts |
| `--wait` | Wait for stack deletion to complete |
| `--cleanup` | Automatically empty S3 buckets and ECR repositories (default: enabled) |
| `--no-cleanup` | Skip automatic cleanup (manual cleanup required) |
| `--list` | Show what would be removed without actually removing (dry run) |

#### How It Works

1. **Application Stacks**: Uses standard `cdk destroy` command (CDK best practice)
   - Automatically handles dependencies between stacks
   - Respects CDK removal policies
   - Can be run independently: `cdk destroy --all`

2. **Bootstrap Stack**: Handled separately with automatic cleanup
   - Empties S3 buckets before deletion
   - Deletes ECR images before repository removal
   - Prevents deletion failures due to non-empty resources

#### Examples

```bash
# Remove all application infrastructure (keeps bootstrap)
./cdk-teardown --all

# Remove dev environment only
./cdk-teardown --environment dev

# Complete teardown (application + bootstrap)
./cdk-teardown --all --bootstrap

# Remove without confirmation
./cdk-teardown --all --bootstrap --force

# Check what would be removed first
./cdk-teardown --list
./cdk-teardown --list --bootstrap
```

#### Troubleshooting

**API Gateway still exists after teardown?**
- Make sure you're removing the application stacks, not just bootstrap
- Use `./cdk-teardown --all` to remove application stacks
- Bootstrap removal (`--bootstrap`) only removes CDKToolkit, not your app

**Resources fail to delete?**
- The script automatically empties S3/ECR, but if issues persist:
  - Check CloudFormation console for failed resources
  - Use `bootstrap_utils.py resource-details` to get specific error info
  - Manually clean up resources if needed

> **⚠️ Important**: 
> - Removing bootstrap (`--bootstrap`) will prevent future CDK deployments until you run `cdk bootstrap` again
> - Application stacks can be recreated with `cdk deploy --all`
> - Bootstrap is shared across projects - only remove if you're sure no other CDK projects need it

> **💡 Tip**: After running `cdk bootstrap`, you can use `python bootstrap_utils.py list` to see what resources were created. See the [Manage CDK Bootstrap Resources](#manage-cdk-bootstrap-resources) section below for more details.

### Manage CDK Bootstrap Resources

The `cdk bootstrap` command creates a CloudFormation stack (`CDKToolkit`) containing essential resources needed for CDK deployments (S3 buckets for assets, IAM roles, ECR repositories, etc.). This project includes utilities to help you list, inspect, and manage these bootstrap resources.

> **Note**: These utilities are particularly useful for:
> - Auditing what resources were created by bootstrap
> - Cleaning up bootstrap resources when no longer needed
> - Managing multiple bootstrap stacks across regions or with custom qualifiers

#### Quick Reference

| Command | Description | Example |
|---------|-------------|---------|
| `list` | List all resources in bootstrap stack | `python bootstrap_utils.py list` |
| `list-all` | List all bootstrap stacks in region | `python bootstrap_utils.py list-all` |
| `list-qualifiers` | List all qualifiers used | `python bootstrap_utils.py list-qualifiers` |
| `resource-details` | Get details about a specific resource | `python bootstrap_utils.py resource-details ResourceName` |
| `remove` | Delete bootstrap stack | `python bootstrap_utils.py remove --wait` |

#### List Bootstrap Resources

```bash
# List resources in the default bootstrap stack
cd infrastructure
python bootstrap_utils.py list

# List resources in a specific region
python bootstrap_utils.py list --region us-west-2

# List resources with custom qualifier
python bootstrap_utils.py list --qualifier my-qualifier

# Output as JSON
python bootstrap_utils.py list --format json
```

#### List All Bootstrap Stacks

```bash
# List all bootstrap stacks in the region
python bootstrap_utils.py list-all

# List all bootstrap stacks in a specific region
python bootstrap_utils.py list-all --region us-west-2
```

#### Remove Bootstrap Stack

The removal process automatically empties S3 buckets and ECR repositories before deleting the stack to prevent deletion failures:

```bash
# Remove the default bootstrap stack (with automatic cleanup)
cd infrastructure
python bootstrap_utils.py remove

# Remove without confirmation prompt
python bootstrap_utils.py remove --force

# Remove and wait for completion
python bootstrap_utils.py remove --wait

# Remove bootstrap stack with custom qualifier
python bootstrap_utils.py remove --qualifier my-qualifier

# Remove without automatic cleanup (manual cleanup required)
python bootstrap_utils.py remove --no-cleanup
```

**Automatic Cleanup Features:**
- ✅ **S3 Buckets**: Automatically deletes all objects, versions, and delete markers
- ✅ **ECR Repositories**: Automatically deletes all container images
- ✅ **Error Handling**: Continues with deletion even if some cleanup operations fail
- ✅ **Detailed Reporting**: Shows cleanup results for each resource

The cleanup happens automatically before stack deletion to ensure resources can be removed successfully.

#### Using in Python

You can also use these functions programmatically:

```python
from infrastructure.stack import (
    list_bootstrap_resources,
    remove_bootstrap_stack,
    list_all_bootstrap_stacks,
)

# List bootstrap resources
resources = list_bootstrap_resources(region="us-east-1")
print(f"Found {len(resources['resources'])} resources")

# List all bootstrap stacks
stacks = list_all_bootstrap_stacks(region="us-east-1")
for stack in stacks:
    print(f"{stack['StackName']}: {stack['StackStatus']}")

# Remove bootstrap stack with automatic cleanup (default)
result = remove_bootstrap_stack(region="us-east-1", wait=True, cleanup_resources=True)
print(result['message'])

# Check cleanup results
if result.get('cleanup_results'):
    cleanup = result['cleanup_results']
    print(f"S3 buckets emptied: {len([r for r in cleanup.get('s3_buckets', []) if r.get('status') == 'success'])}")
    print(f"ECR repositories emptied: {len([r for r in cleanup.get('ecr_repositories', []) if r.get('status') == 'success'])}")

# Remove without automatic cleanup
result = remove_bootstrap_stack(region="us-east-1", cleanup_resources=False)
```

#### Handling Failed or Skipped Resources

When listing bootstrap resources, you may encounter resources with statuses like `DELETE_FAILED`, `CREATE_FAILED`, or `SKIP`. The utilities provide enhanced handling for these cases:

```bash
# List resources - failed/skipped resources are highlighted
python bootstrap_utils.py list

# Get detailed information about a specific failed resource
python bootstrap_utils.py resource-details ContainerAssetsRepository

# Get details with qualifier
python bootstrap_utils.py resource-details ContainerAssetsRepository --qualifier prod
```

The `resource-details` command provides:
- Detailed status information
- Failure reasons
- Specific recommendations based on resource type

**Common Issues and Solutions:**

> **💡 Note**: The `remove` command now automatically handles cleanup of S3 buckets and ECR repositories. The manual steps below are only needed if you use `--no-cleanup` or encounter issues.

1. **DELETE_FAILED on ECR Repository**: Repository may contain images
   - **Automatic**: Handled automatically by `remove` command (default behavior)
   - **Manual**: If using `--no-cleanup` or if automatic cleanup fails:
   ```bash
   # Get the repository name from resource details, then:
   aws ecr list-images --repository-name <repo-name> --query 'imageIds[*]' --output json | \
     aws ecr batch-delete-image --repository-name <repo-name> --image-ids file:///dev/stdin
   # Or force delete:
   aws ecr delete-repository --repository-name <repo-name> --force
   ```

2. **DELETE_FAILED on S3 Bucket**: Bucket may not be empty
   - **Automatic**: Handled automatically by `remove` command (default behavior)
   - **Manual**: If using `--no-cleanup` or if automatic cleanup fails:
   ```bash
   # Empty the bucket first:
   aws s3 rm s3://<bucket-name> --recursive
   # Or use:
   aws s3 rb s3://<bucket-name> --force
   ```

3. **After fixing resources**: Retry stack deletion or continue with CloudFormation console.

**Using in Python:**

```python
from infrastructure.stack import get_failed_resource_details

# Get detailed information about a failed resource
details = get_failed_resource_details(
    logical_resource_id="ContainerAssetsRepository",
    region="us-east-1"
)

print(f"Status: {details['ResourceStatus']}")
print(f"Reason: {details['ResourceStatusReason']}")
print("Recommendations:")
for rec in details['Recommendations']:
    print(f"  - {rec}")
```

> **⚠️ Warning**: Removing the bootstrap stack will prevent future CDK deployments until you run `cdk bootstrap` again. Make sure no active CDK stacks depend on the bootstrap resources before removing them.

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
│   ├── stack.py              # CDK stacks & bootstrap utilities
│   ├── bootstrap_utils.py    # CLI for bootstrap management
│   ├── cdk-teardown          # Teardown script (opposite of cdk bootstrap)
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

