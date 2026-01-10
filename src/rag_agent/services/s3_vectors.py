"""
S3 Vector Storage Service.

Handles vector storage and retrieval using Amazon S3 Vectors
for cost-effective RAG applications.
"""

import json
import structlog
from typing import Any
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_agent.config import Settings, get_settings
from rag_agent.models import DocumentChunk, VectorSearchResult

logger = structlog.get_logger(__name__)


class S3VectorService:
    """
    Service for managing vector storage in Amazon S3.
    
    Uses S3 Vectors feature for efficient vector storage and similarity search,
    integrated with Amazon Bedrock Knowledge Bases.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize S3 Vector Service."""
        self.settings = settings or get_settings()
        self._s3_client = None
        self._s3_vectors_client = None

    def _get_boto_kwargs(self) -> dict:
        """Get boto3 client kwargs, only including credentials if explicitly set."""
        kwargs = {"region_name": self.settings.aws.region}
        if self.settings.aws.access_key_id and self.settings.aws.secret_access_key:
            kwargs["aws_access_key_id"] = self.settings.aws.access_key_id
            kwargs["aws_secret_access_key"] = self.settings.aws.secret_access_key
        return kwargs

    @property
    def s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            self._s3_client = boto3.client("s3", **self._get_boto_kwargs())
        return self._s3_client

    @property
    def s3_vectors_client(self):
        """Lazy initialization of S3 Vectors client."""
        if self._s3_vectors_client is None:
            # S3 Vectors uses the S3 API with vector-specific operations
            self._s3_vectors_client = boto3.client("s3vectors", **self._get_boto_kwargs())
        return self._s3_vectors_client

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_vector_index(self, index_name: str | None = None) -> dict[str, Any]:
        """
        Create a vector index in S3 Vectors.
        
        Args:
            index_name: Name for the vector index
            
        Returns:
            Index creation response
        """
        index_name = index_name or self.settings.s3.vector_index_name
        
        try:
            response = self.s3_vectors_client.create_index(
                vectorBucketName=self.settings.s3.bucket_name,
                indexName=index_name,
                dimension=self.settings.vector.dimension,
                distanceMetric="cosine",
            )
            logger.info("Vector index created", index_name=index_name)
            return response
        except ClientError as e:
            if e.response["Error"]["Code"] == "IndexAlreadyExists":
                logger.info("Vector index already exists", index_name=index_name)
                return {"status": "exists", "indexName": index_name}
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def store_vectors(
        self,
        chunks: list[DocumentChunk],
        document_id: UUID,
    ) -> dict[str, Any]:
        """
        Store document chunk vectors in S3 Vectors.
        
        Args:
            chunks: List of document chunks with embeddings
            document_id: Parent document ID
            
        Returns:
            Storage operation results
        """
        vectors_to_store = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(
                    "Skipping chunk without embedding",
                    chunk_id=str(chunk.metadata.chunk_id),
                )
                continue
                
            vector_data = {
                "key": str(chunk.metadata.chunk_id),
                "data": {
                    "vector": chunk.embedding,
                },
                "metadata": {
                    "document_id": str(document_id),
                    "chunk_index": chunk.metadata.chunk_index,
                    "content": chunk.content[:1000],  # Store truncated content
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                },
            }
            vectors_to_store.append(vector_data)

        if not vectors_to_store:
            logger.warning("No vectors to store", document_id=str(document_id))
            return {"stored": 0}

        try:
            # Batch upsert vectors
            response = self.s3_vectors_client.put_vectors(
                vectorBucketName=self.settings.s3.bucket_name,
                indexName=self.settings.s3.vector_index_name,
                vectors=vectors_to_store,
            )
            
            logger.info(
                "Vectors stored successfully",
                document_id=str(document_id),
                vector_count=len(vectors_to_store),
            )
            return {"stored": len(vectors_to_store), "response": response}
            
        except ClientError as e:
            logger.error(
                "Failed to store vectors",
                document_id=str(document_id),
                error=str(e),
            )
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_vectors(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        top_k = top_k or self.settings.vector.top_k
        
        try:
            search_params = {
                "vectorBucketName": self.settings.s3.bucket_name,
                "indexName": self.settings.s3.vector_index_name,
                "queryVector": query_vector,
                "topK": top_k,
            }
            
            if filters:
                search_params["filter"] = filters

            response = self.s3_vectors_client.query_vectors(**search_params)
            
            results = []
            for match in response.get("matches", []):
                score = match.get("score", 0.0)
                
                # Filter by similarity threshold
                if score < self.settings.vector.similarity_threshold:
                    continue
                    
                metadata = match.get("metadata", {})
                results.append(
                    VectorSearchResult(
                        document_id=metadata.get("document_id", ""),
                        chunk_id=match.get("key", ""),
                        score=score,
                        content=metadata.get("content", ""),
                        metadata=metadata,
                    )
                )
            
            logger.info(
                "Vector search completed",
                results_count=len(results),
                top_k=top_k,
            )
            return results
            
        except ClientError as e:
            logger.error("Vector search failed", error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def delete_vectors(
        self,
        vector_ids: list[str],
    ) -> dict[str, Any]:
        """
        Delete vectors by their IDs.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            Deletion operation results
        """
        try:
            response = self.s3_vectors_client.delete_vectors(
                vectorBucketName=self.settings.s3.bucket_name,
                indexName=self.settings.s3.vector_index_name,
                keys=vector_ids,
            )
            
            logger.info("Vectors deleted", count=len(vector_ids))
            return {"deleted": len(vector_ids), "response": response}
            
        except ClientError as e:
            logger.error("Failed to delete vectors", error=str(e))
            raise

    async def upload_document(
        self,
        document_id: str,
        content: bytes,
        content_type: str = "application/json",
    ) -> str:
        """
        Upload a document to S3 for storage.
        
        Args:
            document_id: Unique document identifier
            content: Document content as bytes
            content_type: MIME type of the content
            
        Returns:
            S3 key of the uploaded document
        """
        s3_key = f"{self.settings.s3.documents_prefix}{document_id}"
        
        try:
            self.s3_client.put_object(
                Bucket=self.settings.s3.bucket_name,
                Key=s3_key,
                Body=content,
                ContentType=content_type,
            )
            logger.info("Document uploaded to S3", document_id=document_id, s3_key=s3_key)
            return s3_key
            
        except ClientError as e:
            logger.error("Failed to upload document", document_id=document_id, error=str(e))
            raise

    async def get_document(self, document_id: str) -> bytes | None:
        """
        Retrieve a document from S3.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Document content as bytes or None if not found
        """
        s3_key = f"{self.settings.s3.documents_prefix}{document_id}"
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.settings.s3.bucket_name,
                Key=s3_key,
            )
            return response["Body"].read()
            
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.warning("Document not found", document_id=document_id)
                return None
            raise

