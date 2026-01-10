"""
RAG Agent - AI Agent with AWS Bedrock and Google ADK patterns.

A production-ready RAG (Retrieval-Augmented Generation) application that leverages
AWS Bedrock services, S3 vector storage, DynamoDB, and Google's agent development patterns.
"""

__version__ = "0.1.0"
__author__ = "Developer"

from rag_agent.config import Settings
from rag_agent.agent import RAGAgent

__all__ = ["Settings", "RAGAgent", "__version__"]

