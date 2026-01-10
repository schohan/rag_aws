"""
Embedding Service for generating vector embeddings.

Uses Amazon Bedrock's embedding models to convert text into vector representations.
"""

import json
from typing import Any

import boto3
import structlog
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_agent.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using Amazon Bedrock.
    
    Supports various embedding models including Amazon Titan and Cohere.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize Embedding Service."""
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Bedrock Runtime client."""
        if self._client is None:
            # Only pass credentials if explicitly set (for local dev)
            # In Lambda, use IAM role credentials automatically
            kwargs = {"region_name": self.settings.aws.region}
            if self.settings.aws.access_key_id and self.settings.aws.secret_access_key:
                kwargs["aws_access_key_id"] = self.settings.aws.access_key_id
                kwargs["aws_secret_access_key"] = self.settings.aws.secret_access_key
            self._client = boto3.client("bedrock-runtime", **kwargs)
        return self._client

    def _get_embedding_request_body(self, text: str, model_id: str) -> dict[str, Any]:
        """
        Get the appropriate request body format for the embedding model.
        
        Args:
            text: Text to embed
            model_id: Model identifier
            
        Returns:
            Request body dictionary
        """
        if "titan" in model_id.lower():
            return {
                "inputText": text,
                "dimensions": self.settings.vector.dimension,
                "normalize": True,
            }
        elif "cohere" in model_id.lower():
            return {
                "texts": [text],
                "input_type": "search_document",
                "truncate": "END",
            }
        else:
            # Default format
            return {"inputText": text}

    def _parse_embedding_response(
        self,
        response: dict[str, Any],
        model_id: str,
    ) -> list[float]:
        """
        Parse the embedding from the model response.
        
        Args:
            response: Model response
            model_id: Model identifier
            
        Returns:
            Embedding vector
        """
        if "titan" in model_id.lower():
            return response.get("embedding", [])
        elif "cohere" in model_id.lower():
            embeddings = response.get("embeddings", [[]])
            return embeddings[0] if embeddings else []
        else:
            return response.get("embedding", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_embedding(
        self,
        text: str,
        model_id: str | None = None,
    ) -> list[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            model_id: Optional model ID override
            
        Returns:
            Embedding vector as list of floats
        """
        model_id = model_id or self.settings.bedrock.embedding_model_id
        
        # Truncate text if too long (most models have ~8k token limit)
        max_chars = 25000  # Conservative estimate
        if len(text) > max_chars:
            logger.warning(
                "Text truncated for embedding",
                original_length=len(text),
                truncated_length=max_chars,
            )
            text = text[:max_chars]

        request_body = self._get_embedding_request_body(text, model_id)
        
        try:
            response = self.client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )
            
            response_body = json.loads(response["body"].read())
            embedding = self._parse_embedding_response(response_body, model_id)
            
            logger.debug(
                "Embedding generated",
                text_length=len(text),
                embedding_dimension=len(embedding),
            )
            return embedding
            
        except ClientError as e:
            logger.error(
                "Failed to generate embedding",
                model_id=model_id,
                error=str(e),
            )
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def generate_embeddings_batch(
        self,
        texts: list[str],
        model_id: str | None = None,
        batch_size: int = 10,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Note: Most Bedrock embedding models don't support true batching,
        so this processes texts sequentially but with retry logic.
        
        Args:
            texts: List of texts to embed
            model_id: Optional model ID override
            batch_size: Number of texts to process in parallel (for future optimization)
            
        Returns:
            List of embedding vectors
        """
        model_id = model_id or self.settings.bedrock.embedding_model_id
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = await self.generate_embedding(text, model_id)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(
                        "Batch embedding progress",
                        completed=i + 1,
                        total=len(texts),
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to generate embedding for text",
                    index=i,
                    error=str(e),
                )
                # Add empty embedding to maintain index alignment
                embeddings.append([])
        
        logger.info(
            "Batch embeddings completed",
            total=len(texts),
            successful=sum(1 for e in embeddings if e),
        )
        return embeddings

    async def generate_query_embedding(
        self,
        query: str,
        model_id: str | None = None,
    ) -> list[float]:
        """
        Generate an embedding optimized for search queries.
        
        Some models (like Cohere) have different input types for queries vs documents.
        
        Args:
            query: Search query text
            model_id: Optional model ID override
            
        Returns:
            Query embedding vector
        """
        model_id = model_id or self.settings.bedrock.embedding_model_id
        
        # For Cohere models, use search_query input type
        if "cohere" in model_id.lower():
            request_body = {
                "texts": [query],
                "input_type": "search_query",
                "truncate": "END",
            }
            
            try:
                response = self.client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body),
                )
                
                response_body = json.loads(response["body"].read())
                embeddings = response_body.get("embeddings", [[]])
                return embeddings[0] if embeddings else []
                
            except ClientError as e:
                logger.error(
                    "Failed to generate query embedding",
                    model_id=model_id,
                    error=str(e),
                )
                raise
        else:
            # For other models, use the standard embedding method
            return await self.generate_embedding(query, model_id)

