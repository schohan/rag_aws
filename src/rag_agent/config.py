"""
Configuration settings for the RAG Agent application.

Uses pydantic-settings for type-safe configuration management with
environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSSettings(BaseSettings):
    """AWS-specific configuration settings."""

    model_config = SettingsConfigDict(extra="ignore")

    # Use explicit env var names to avoid capturing Lambda's temporary credentials
    region: str = Field(default="us-east-1", alias="AWS_DEFAULT_REGION")
    access_key_id: str | None = Field(default=None, alias="RAG_AWS_ACCESS_KEY_ID")
    secret_access_key: str | None = Field(default=None, alias="RAG_AWS_SECRET_ACCESS_KEY")


class S3Settings(BaseSettings):
    """S3 vector storage configuration."""

    model_config = SettingsConfigDict(env_prefix="S3_", extra="ignore")

    bucket_name: str = Field(default="rag-agent-vectors", description="S3 bucket name")
    vector_index_name: str = Field(default="rag-vectors", description="S3 vector index name")
    documents_prefix: str = Field(default="documents/", description="S3 prefix for documents")
    vectors_prefix: str = Field(default="vectors/", description="S3 prefix for vectors")


class DynamoDBSettings(BaseSettings):
    """DynamoDB configuration."""

    model_config = SettingsConfigDict(env_prefix="DYNAMODB_", extra="ignore")

    table_name: str = Field(default="rag-agent-metadata", description="DynamoDB table name")
    read_capacity: int = Field(default=5, description="Read capacity units")
    write_capacity: int = Field(default=5, description="Write capacity units")


class BedrockSettings(BaseSettings):
    """AWS Bedrock configuration."""

    model_config = SettingsConfigDict(env_prefix="BEDROCK_", extra="ignore")

    # Knowledge base and agent IDs
    knowledge_base_id: str | None = Field(default=None, description="Knowledge base ID")
    agent_id: str | None = Field(default=None, description="Bedrock agent ID")
    
    # Model IDs - change these to use different models
    # Supported: Claude, Llama, Titan, Mistral, Cohere, AI21
    embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Embedding model ID (Titan, Cohere)",
    )
    llm_model_id: str = Field(
        default="qwen.qwen3-32b-v1:0",
        description="LLM model ID for generation",
    )
    agent_model_id: str = Field(
        default="qwen.qwen3-32b-v1:0",
        description="Model ID for agent reasoning",
    )
    
    # Model-specific API versions (used when applicable)
    anthropic_version: str = Field(
        default="bedrock-2023-05-31",
        description="Anthropic API version for Claude models",
    )
    
    # Generation parameters (applied based on model support)
    max_tokens: int = Field(default=4096, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=250, description="Top-k sampling parameter")
    
    # Stop sequences (optional)
    stop_sequences: list[str] = Field(
        default_factory=list,
        description="Stop sequences for text generation",
    )


class GoogleAISettings(BaseSettings):
    """Google AI configuration for hybrid agent capabilities."""

    model_config = SettingsConfigDict(env_prefix="GOOGLE_", extra="ignore")

    api_key: str | None = Field(default=None, description="Google AI API key")
    model_name: str = Field(default="gemini-1.5-pro", description="Google model name")


class VectorSettings(BaseSettings):
    """Vector search configuration."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_", extra="ignore")

    dimension: int = Field(default=1024, description="Vector dimension")
    top_k: int = Field(default=5, alias="TOP_K_RESULTS", description="Number of results to return")
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity threshold",
    )

    @field_validator("similarity_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")

    size: int = Field(default=1000, description="Chunk size in characters")
    overlap: int = Field(default=200, description="Chunk overlap in characters")

    @field_validator("overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        if "size" in info.data and v >= info.data["size"]:
            raise ValueError("Overlap must be less than chunk size")
        return v


class Settings(BaseSettings):
    """Main application settings aggregating all configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application settings
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        alias="APP_ENV",
    )
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Nested settings
    aws: AWSSettings = Field(default_factory=AWSSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    dynamodb: DynamoDBSettings = Field(default_factory=DynamoDBSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    google: GoogleAISettings = Field(default_factory=GoogleAISettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

