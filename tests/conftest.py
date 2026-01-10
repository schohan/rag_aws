"""
Pytest configuration and fixtures for RAG Agent tests.
"""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock

# Set test environment variables
os.environ["APP_ENV"] = "development"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["S3_BUCKET_NAME"] = "test-bucket"
os.environ["DYNAMODB_TABLE_NAME"] = "test-table"


@pytest.fixture
def settings():
    """Create test settings."""
    from rag_agent.config import Settings
    return Settings()


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client."""
    client = MagicMock()
    client.invoke_model = MagicMock(return_value={
        "body": MagicMock(read=lambda: b'{"content": [{"text": "Test response"}], "usage": {"input_tokens": 10, "output_tokens": 20}}')
    })
    return client


@pytest.fixture
def mock_dynamodb_table():
    """Create a mock DynamoDB table."""
    table = MagicMock()
    table.put_item = MagicMock(return_value={})
    table.get_item = MagicMock(return_value={"Item": None})
    table.query = MagicMock(return_value={"Items": []})
    return table


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = MagicMock()
    client.put_object = MagicMock(return_value={})
    client.get_object = MagicMock(return_value={
        "Body": MagicMock(read=lambda: b"Test content")
    })
    return client


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    from rag_agent.models import Document
    return Document(
        title="Test Document",
        content="This is a test document with some content for testing the RAG agent.",
        source="test",
        metadata={"test": True},
    )


@pytest.fixture
def sample_query_request():
    """Create a sample query request."""
    from rag_agent.models import QueryRequest
    return QueryRequest(
        query="What is the test document about?",
        top_k=3,
    )


@pytest.fixture
async def agent(settings, mock_bedrock_client):
    """Create a RAG agent with mocked services."""
    from rag_agent.agent import RAGAgent
    
    agent = RAGAgent(settings=settings)
    # Mock the bedrock service
    agent.bedrock._runtime_client = mock_bedrock_client
    
    return agent

