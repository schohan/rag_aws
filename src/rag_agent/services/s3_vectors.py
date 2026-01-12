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

    async def create_s3_bucket(self) -> dict[str, Any]:
        """
        Create the regular S3 bucket for document storage.
        
        This is separate from the vector bucket and is used for storing
        document content files.
        
        Returns:
            Bucket creation response
        """
        try:
            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=self.settings.s3.bucket_name)
                logger.info("S3 bucket already exists", bucket_name=self.settings.s3.bucket_name)
                return {"status": "exists", "bucketName": self.settings.s3.bucket_name}
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise
            
            # Create bucket if it doesn't exist
            create_kwargs = {"Bucket": self.settings.s3.bucket_name}
            
            # Set region for bucket creation
            if self.settings.aws.region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.settings.aws.region
                }
            
            self.s3_client.create_bucket(**create_kwargs)
            logger.info("S3 bucket created", bucket_name=self.settings.s3.bucket_name)
            return {"status": "created", "bucketName": self.settings.s3.bucket_name}
            
        except ClientError as e:
            if e.response["Error"]["Code"] in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
                logger.info("S3 bucket already exists", bucket_name=self.settings.s3.bucket_name)
                return {"status": "exists", "bucketName": self.settings.s3.bucket_name}
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_vector_bucket(self) -> dict[str, Any]:
        """
        Create a vector bucket in S3 Vectors.
        
        The vector bucket must exist before creating an index.
        This is separate from the regular S3 bucket.
        
        Returns:
            Bucket creation response
        """
        try:
            response = self.s3_vectors_client.create_vector_bucket(
                vectorBucketName=self.settings.s3.bucket_name,
            )
            logger.info("Vector bucket created", bucket_name=self.settings.s3.bucket_name)
            return response
        except ClientError as e:
            if e.response["Error"]["Code"] in ("BucketAlreadyExists", "ConflictException"):
                logger.info("Vector bucket already exists", bucket_name=self.settings.s3.bucket_name)
                return {"status": "exists", "bucketName": self.settings.s3.bucket_name}
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def create_vector_index(self, index_name: str | None = None) -> dict[str, Any]:
        """
        Create a vector index in S3 Vectors.
        
        The vector bucket must exist before creating an index.
        
        Args:
            index_name: Name for the vector index
            
        Returns:
            Index creation response
        """
        index_name = index_name or self.settings.s3.vector_index_name
        
        # Ensure vector bucket exists first
        try:
            await self.create_vector_bucket()
        except Exception as e:
            logger.warning("Could not ensure vector bucket exists", error=str(e))
        
        try:
            response = self.s3_vectors_client.create_index(
                vectorBucketName=self.settings.s3.bucket_name,
                indexName=index_name,
                dataType="float32",  # Required parameter for S3 Vectors
                dimension=self.settings.vector.dimension,
                distanceMetric="cosine",
            )
            logger.info("Vector index created", index_name=index_name)
            return response
        except ClientError as e:
            if e.response["Error"]["Code"] in ("IndexAlreadyExists", "ConflictException"):
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
                
            # S3 Vectors API expects float32 directly in data field
            vector_data = {
                "key": str(chunk.metadata.chunk_id),
                "data": {
                    "float32": chunk.embedding,
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
            # S3 Vectors API expects queryVector with "float32" key
            search_params = {
                "vectorBucketName": self.settings.s3.bucket_name,
                "indexName": self.settings.s3.vector_index_name,
                "queryVector": {"float32": query_vector},
                "topK": top_k,
            }
            
            if filters:
                search_params["filter"] = filters

            response = self.s3_vectors_client.query_vectors(**search_params)
            
            all_matches = response.get("matches", [])
            logger.info(
                "Vector search API response",
                total_matches=len(all_matches),
                threshold=self.settings.vector.similarity_threshold,
            )
            
            results = []
            filtered_count = 0
            for match in all_matches:
                score = match.get("score", 0.0)
                
                # Log all scores for debugging
                logger.debug(
                    "Match score",
                    score=score,
                    threshold=self.settings.vector.similarity_threshold,
                    above_threshold=score >= self.settings.vector.similarity_threshold,
                )
                
                # Filter by similarity threshold
                if score < self.settings.vector.similarity_threshold:
                    filtered_count += 1
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
                filtered_count=filtered_count,
                total_matches=len(all_matches),
                top_k=top_k,
                threshold=self.settings.vector.similarity_threshold,
            )
            
            # If no results but we have matches, log warning and return top results anyway
            if len(results) == 0 and len(all_matches) > 0:
                logger.warning(
                    "All results filtered by threshold, returning top match anyway",
                    top_score=all_matches[0].get("score", 0.0) if all_matches else 0.0,
                    threshold=self.settings.vector.similarity_threshold,
                )
                # Return at least the top match even if below threshold
                top_match = all_matches[0]
                metadata = top_match.get("metadata", {})
                results.append(
                    VectorSearchResult(
                        document_id=metadata.get("document_id", ""),
                        chunk_id=top_match.get("key", ""),
                        score=top_match.get("score", 0.0),
                        content=metadata.get("content", ""),
                        metadata=metadata,
                    )
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

