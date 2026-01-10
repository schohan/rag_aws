"""
DynamoDB Service for metadata storage.

Handles document metadata, conversation history, and agent state persistence.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import boto3
import structlog
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_agent.config import Settings, get_settings
from rag_agent.models import DocumentMetadata, DocumentStatus, Conversation

logger = structlog.get_logger(__name__)


class DynamoDBService:
    """
    Service for managing metadata in Amazon DynamoDB.
    
    Stores document metadata, conversation history, and agent execution logs.
    """

    def __init__(self, settings: Settings | None = None):
        """Initialize DynamoDB Service."""
        self.settings = settings or get_settings()
        self._dynamodb = None
        self._table = None

    @property
    def dynamodb(self):
        """Lazy initialization of DynamoDB resource."""
        if self._dynamodb is None:
            # Only pass credentials if explicitly set (for local dev)
            # In Lambda, use IAM role credentials automatically
            kwargs = {"region_name": self.settings.aws.region}
            if self.settings.aws.access_key_id and self.settings.aws.secret_access_key:
                kwargs["aws_access_key_id"] = self.settings.aws.access_key_id
                kwargs["aws_secret_access_key"] = self.settings.aws.secret_access_key
            self._dynamodb = boto3.resource("dynamodb", **kwargs)
        return self._dynamodb

    @property
    def table(self):
        """Get the DynamoDB table."""
        if self._table is None:
            self._table = self.dynamodb.Table(self.settings.dynamodb.table_name)
        return self._table

    async def create_table_if_not_exists(self) -> bool:
        """
        Create the DynamoDB table if it doesn't exist.
        
        Returns:
            True if table was created, False if it already exists
        """
        try:
            # Check if table exists
            self.dynamodb.meta.client.describe_table(
                TableName=self.settings.dynamodb.table_name
            )
            logger.info("DynamoDB table already exists", table=self.settings.dynamodb.table_name)
            return False
            
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Create the table
                table = self.dynamodb.create_table(
                    TableName=self.settings.dynamodb.table_name,
                    KeySchema=[
                        {"AttributeName": "PK", "KeyType": "HASH"},
                        {"AttributeName": "SK", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "PK", "AttributeType": "S"},
                        {"AttributeName": "SK", "AttributeType": "S"},
                        {"AttributeName": "GSI1PK", "AttributeType": "S"},
                        {"AttributeName": "GSI1SK", "AttributeType": "S"},
                    ],
                    GlobalSecondaryIndexes=[
                        {
                            "IndexName": "GSI1",
                            "KeySchema": [
                                {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                                {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                            ],
                            "Projection": {"ProjectionType": "ALL"},
                            "ProvisionedThroughput": {
                                "ReadCapacityUnits": self.settings.dynamodb.read_capacity,
                                "WriteCapacityUnits": self.settings.dynamodb.write_capacity,
                            },
                        }
                    ],
                    ProvisionedThroughput={
                        "ReadCapacityUnits": self.settings.dynamodb.read_capacity,
                        "WriteCapacityUnits": self.settings.dynamodb.write_capacity,
                    },
                )
                table.wait_until_exists()
                logger.info("DynamoDB table created", table=self.settings.dynamodb.table_name)
                return True
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def save_document_metadata(self, metadata: DocumentMetadata) -> dict[str, Any]:
        """
        Save document metadata to DynamoDB.
        
        Args:
            metadata: Document metadata to save
            
        Returns:
            DynamoDB response
        """
        item = {
            "PK": f"DOC#{metadata.id}",
            "SK": "METADATA",
            "GSI1PK": f"STATUS#{metadata.status.value}",
            "GSI1SK": metadata.created_at,
            "id": metadata.id,
            "title": metadata.title,
            "source": metadata.source,
            "content_type": metadata.content_type,
            "status": metadata.status.value,
            "chunk_count": metadata.chunk_count,
            "created_at": metadata.created_at,
            "updated_at": metadata.updated_at,
            "entity_type": "DOCUMENT",
        }
        
        if metadata.token_count:
            item["token_count"] = metadata.token_count
        if metadata.indexed_at:
            item["indexed_at"] = metadata.indexed_at
        if metadata.s3_key:
            item["s3_key"] = metadata.s3_key
        if metadata.extra_metadata:
            item["extra_metadata"] = metadata.extra_metadata

        try:
            response = self.table.put_item(Item=item)
            logger.info("Document metadata saved", document_id=metadata.id)
            return response
            
        except ClientError as e:
            logger.error("Failed to save document metadata", document_id=metadata.id, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_document_metadata(self, document_id: str) -> DocumentMetadata | None:
        """
        Retrieve document metadata from DynamoDB.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Document metadata or None if not found
        """
        try:
            response = self.table.get_item(
                Key={
                    "PK": f"DOC#{document_id}",
                    "SK": "METADATA",
                }
            )
            
            item = response.get("Item")
            if not item:
                return None
                
            return DocumentMetadata(
                id=item["id"],
                title=item["title"],
                source=item["source"],
                content_type=item["content_type"],
                status=DocumentStatus(item["status"]),
                chunk_count=item.get("chunk_count", 0),
                token_count=item.get("token_count"),
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                indexed_at=item.get("indexed_at"),
                s3_key=item.get("s3_key"),
                extra_metadata=item.get("extra_metadata", {}),
            )
            
        except ClientError as e:
            logger.error("Failed to get document metadata", document_id=document_id, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        indexed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Update document processing status.
        
        Args:
            document_id: Unique document identifier
            status: New status
            indexed_at: Timestamp when document was indexed
            
        Returns:
            DynamoDB response
        """
        update_expression = "SET #status = :status, updated_at = :updated_at, GSI1PK = :gsi1pk"
        expression_values = {
            ":status": status.value,
            ":updated_at": datetime.utcnow().isoformat(),
            ":gsi1pk": f"STATUS#{status.value}",
        }
        
        if indexed_at:
            update_expression += ", indexed_at = :indexed_at"
            expression_values[":indexed_at"] = indexed_at.isoformat()

        try:
            response = self.table.update_item(
                Key={
                    "PK": f"DOC#{document_id}",
                    "SK": "METADATA",
                },
                UpdateExpression=update_expression,
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=expression_values,
                ReturnValues="ALL_NEW",
            )
            logger.info("Document status updated", document_id=document_id, status=status.value)
            return response
            
        except ClientError as e:
            logger.error(
                "Failed to update document status",
                document_id=document_id,
                error=str(e),
            )
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def list_documents_by_status(
        self,
        status: DocumentStatus,
        limit: int = 50,
    ) -> list[DocumentMetadata]:
        """
        List documents by their processing status.
        
        Args:
            status: Document status to filter by
            limit: Maximum number of results
            
        Returns:
            List of document metadata
        """
        try:
            response = self.table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("GSI1PK").eq(f"STATUS#{status.value}"),
                Limit=limit,
            )
            
            documents = []
            for item in response.get("Items", []):
                documents.append(
                    DocumentMetadata(
                        id=item["id"],
                        title=item["title"],
                        source=item["source"],
                        content_type=item["content_type"],
                        status=DocumentStatus(item["status"]),
                        chunk_count=item.get("chunk_count", 0),
                        token_count=item.get("token_count"),
                        created_at=item["created_at"],
                        updated_at=item["updated_at"],
                        indexed_at=item.get("indexed_at"),
                        s3_key=item.get("s3_key"),
                        extra_metadata=item.get("extra_metadata", {}),
                    )
                )
            
            return documents
            
        except ClientError as e:
            logger.error("Failed to list documents", status=status.value, error=str(e))
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def save_conversation(self, conversation: Conversation) -> dict[str, Any]:
        """
        Save a conversation to DynamoDB.
        
        Args:
            conversation: Conversation to save
            
        Returns:
            DynamoDB response
        """
        item = {
            "PK": f"CONV#{str(conversation.id)}",
            "SK": "METADATA",
            "GSI1PK": "CONVERSATION",
            "GSI1SK": conversation.created_at.isoformat(),
            "id": str(conversation.id),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
                for msg in conversation.messages
            ],
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "metadata": conversation.metadata,
            "entity_type": "CONVERSATION",
        }

        try:
            response = self.table.put_item(Item=item)
            logger.info("Conversation saved", conversation_id=str(conversation.id))
            return response
            
        except ClientError as e:
            logger.error(
                "Failed to save conversation",
                conversation_id=str(conversation.id),
                error=str(e),
            )
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def delete_document(self, document_id: str) -> dict[str, Any]:
        """
        Delete a document and its metadata from DynamoDB.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            DynamoDB response
        """
        try:
            response = self.table.delete_item(
                Key={
                    "PK": f"DOC#{document_id}",
                    "SK": "METADATA",
                }
            )
            logger.info("Document deleted", document_id=document_id)
            return response
            
        except ClientError as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            raise

