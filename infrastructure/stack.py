"""
AWS CDK Stack for RAG Agent Infrastructure.

Creates all necessary AWS resources for the RAG application:
- S3 bucket for document storage
- S3 Vectors for vector embeddings
- DynamoDB table for metadata
- IAM roles and policies
- Bedrock Knowledge Base (optional)
"""

import logging
import os
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from constructs import Construct
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_s3 as s3,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_logs as logs,
)

# Configure logging
logger = logging.getLogger(__name__)

# AWS client configuration with retries
AWS_CONFIG = Config(
    retries={
        "max_attempts": 3,
        "mode": "adaptive",
    },
    region_name=None,  # Will be set per client
)

# Constants
DEFAULT_REGION = "us-east-1"
DEFAULT_STACK_NAME = "CDKToolkit"
STACK_NAME_PREFIX = "CDKToolkit-"
APP_STACK_PREFIXES = ["RAGAgentStack-", "RAGAgentLambdaStack-"]
MAX_WAITER_ATTEMPTS = 120
WAITER_DELAY = 5


class RAGAgentStack(Stack):
    """
    CDK Stack for RAG Agent infrastructure.
    
    Creates S3, DynamoDB, and supporting resources for the RAG application.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str = "dev",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Naming prefix
        prefix = f"rag-agent-{environment}"

        # S3 Bucket for documents and vectors
        self.document_bucket = s3.Bucket(
            self,
            "DocumentBucket",
            bucket_name=f"{prefix}-documents-{self.account}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN if environment == "prod" else RemovalPolicy.DESTROY,
            auto_delete_objects=environment != "prod",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="DeleteOldVersions",
                    noncurrent_version_expiration=Duration.days(30),
                ),
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.PUT, s3.HttpMethods.POST],
                    allowed_origins=["*"],  # Configure for production
                    allowed_headers=["*"],
                ),
            ],
        )

        # DynamoDB Table for metadata
        self.metadata_table = dynamodb.Table(
            self,
            "MetadataTable",
            table_name=f"{prefix}-metadata",
            partition_key=dynamodb.Attribute(
                name="PK",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="SK",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN if environment == "prod" else RemovalPolicy.DESTROY,
            point_in_time_recovery=environment == "prod",
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
        )

        # GSI for status-based queries
        self.metadata_table.add_global_secondary_index(
            index_name="GSI1",
            partition_key=dynamodb.Attribute(
                name="GSI1PK",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="GSI1SK",
                type=dynamodb.AttributeType.STRING,
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # IAM Role for Bedrock access
        self.bedrock_role = iam.Role(
            self,
            "BedrockRole",
            role_name=f"{prefix}-bedrock-role",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            description="Role for Bedrock to access RAG resources",
        )

        # Bedrock policy for model access
        self.bedrock_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockModelAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/*",
                ],
            )
        )

        # S3 access for Bedrock
        self.document_bucket.grant_read(self.bedrock_role)

        # IAM Role for application
        self.app_role = iam.Role(
            self,
            "AppRole",
            role_name=f"{prefix}-app-role",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            ),
            description="Role for RAG Agent application",
        )

        # Grant permissions to app role
        self.document_bucket.grant_read_write(self.app_role)
        self.metadata_table.grant_read_write_data(self.app_role)

        # Bedrock permissions for app
        self.app_role.add_to_policy(
            iam.PolicyStatement(
                sid="BedrockAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:InvokeAgent",
                    "bedrock:Retrieve",
                    "bedrock:RetrieveAndGenerate",
                ],
                resources=["*"],
            )
        )

        # S3 Vectors permissions (preview feature)
        self.app_role.add_to_policy(
            iam.PolicyStatement(
                sid="S3VectorsAccess",
                effect=iam.Effect.ALLOW,
                actions=[
                    "s3vectors:*",
                ],
                resources=[
                    f"arn:aws:s3vectors:{self.region}:{self.account}:*",
                ],
            )
        )

        # CloudWatch Logs permissions
        self.app_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "service-role/AWSLambdaBasicExecutionRole"
            )
        )

        # Outputs
        CfnOutput(
            self,
            "BucketName",
            value=self.document_bucket.bucket_name,
            description="S3 bucket for documents",
            export_name=f"{prefix}-bucket-name",
        )

        CfnOutput(
            self,
            "TableName",
            value=self.metadata_table.table_name,
            description="DynamoDB table for metadata",
            export_name=f"{prefix}-table-name",
        )

        CfnOutput(
            self,
            "AppRoleArn",
            value=self.app_role.role_arn,
            description="IAM role ARN for the application",
            export_name=f"{prefix}-app-role-arn",
        )


class RAGAgentLambdaStack(Stack):
    """
    CDK Stack for Lambda-based RAG Agent deployment.
    
    Deploys the RAG Agent as AWS Lambda functions with API Gateway.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        base_stack: RAGAgentStack,
        environment: str = "dev",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        prefix = f"rag-agent-{environment}"

        # Lambda function for the API using Docker for proper dependency bundling
        self.api_function = lambda_.DockerImageFunction(
            self,
            "ApiFunction",
            function_name=f"{prefix}-api",
            code=lambda_.DockerImageCode.from_image_asset(
                directory="..",  # Project root (parent of infrastructure/)
                file="Dockerfile.lambda",
                exclude=[
                    "cdk.out",
                    "infrastructure/cdk.out",
                    ".venv",
                    "venv",
                    ".git",
                    "__pycache__",
                    "*.pyc",
                    ".pytest_cache",
                    "tests",
                    "docs",
                    ".env",
                    "*.egg-info",
                ],
            ),
            timeout=Duration.seconds(60),
            memory_size=1024,
            role=base_stack.app_role,
            environment={
                "S3_BUCKET_NAME": base_stack.document_bucket.bucket_name,
                "DYNAMODB_TABLE_NAME": base_stack.metadata_table.table_name,
                "LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        # API Gateway
        self.api = apigw.RestApi(
            self,
            "ApiGateway",
            rest_api_name=f"{prefix}-api",
            description="RAG Agent API",
            deploy_options=apigw.StageOptions(
                stage_name=environment,
                throttling_rate_limit=100,
                throttling_burst_limit=200,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        # Lambda integration
        lambda_integration = apigw.LambdaIntegration(
            self.api_function,
            request_templates={"application/json": '{ "statusCode": "200" }'},
        )

        # API routes
        query_resource = self.api.root.add_resource("query")
        query_resource.add_method("POST", lambda_integration)

        chat_resource = self.api.root.add_resource("chat")
        chat_resource.add_method("POST", lambda_integration)

        documents_resource = self.api.root.add_resource("documents")
        documents_resource.add_method("GET", lambda_integration)
        documents_resource.add_method("POST", lambda_integration)

        document_resource = documents_resource.add_resource("{document_id}")
        document_resource.add_method("GET", lambda_integration)
        document_resource.add_method("DELETE", lambda_integration)

        # Outputs
        CfnOutput(
            self,
            "ApiUrl",
            value=self.api.url,
            description="API Gateway URL",
            export_name=f"{prefix}-api-url",
        )


def list_bootstrap_resources(
    region: Optional[str] = None,
    qualifier: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    List all resources created by 'cdk bootstrap' command.
    
    Args:
        region: AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)
        qualifier: CDK bootstrap qualifier (defaults to None for default bootstrap).
                   Qualifiers are alphanumeric strings (with hyphens/underscores allowed)
                   used to create uniquely named bootstrap stacks. When specified during
                   bootstrap with `cdk bootstrap --qualifier <value>`, the stack name
                   becomes `CDKToolkit-<qualifier>`. Use `list_all_bootstrap_stacks()`
                   to discover available qualifiers in your region.
    
    Returns:
        Dictionary containing:
        - 'stack_info': Stack information
        - 'resources': List of resources in the bootstrap stack
        - 'outputs': Stack outputs
    
    Raises:
        ValueError: If the bootstrap stack doesn't exist
        ClientError: If AWS API call fails
    
    Examples:
        # List default bootstrap (no qualifier)
        resources = list_bootstrap_resources(region="us-east-1")
        
        # List bootstrap with qualifier
        resources = list_bootstrap_resources(region="us-east-1", qualifier="prod")
        
        # Find available qualifiers first
        stacks = list_all_bootstrap_stacks(region="us-east-1")
        for stack in stacks:
            if stack['StackName'] != 'CDKToolkit':
                qualifier = stack['StackName'].replace('CDKToolkit-', '')
                print(f"Qualifier: {qualifier}")
    """
    region = region or os.environ.get("CDK_DEFAULT_REGION", DEFAULT_REGION)
    stack_name = f"{STACK_NAME_PREFIX}{qualifier}" if qualifier else DEFAULT_STACK_NAME
    
    # Input validation
    if qualifier and not isinstance(qualifier, str):
        raise ValueError("qualifier must be a string or None")
    if not isinstance(region, str) or not region:
        raise ValueError("region must be a non-empty string")
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    cf_client = boto3.client("cloudformation", region_name=region, config=config)
    
    try:
        # Get stack information
        stack_response = cf_client.describe_stacks(StackName=stack_name)
        stack = stack_response["Stacks"][0]
        
        # Get stack resources
        resources_response = cf_client.list_stack_resources(StackName=stack_name)
        resources = resources_response.get("StackResourceSummaries", [])
        
        # Get detailed resource information for failed/skipped resources
        formatted_resources = []
        for r in resources:
            resource_info = {
                "LogicalResourceId": r["LogicalResourceId"],
                "PhysicalResourceId": r.get("PhysicalResourceId", "N/A"),
                "ResourceType": r["ResourceType"],
                "ResourceStatus": r["ResourceStatus"],
            }
            
            # Add status reason for failed/skipped resources
            if "ResourceStatusReason" in r:
                resource_info["ResourceStatusReason"] = r["ResourceStatusReason"]
            
            # Get detailed resource information if status indicates issues
            status = r.get("ResourceStatus", "")
            if any(keyword in status for keyword in ["FAILED", "SKIP", "DELETE_FAILED", "CREATE_FAILED", "UPDATE_FAILED"]):
                try:
                    # Get detailed resource information
                    resource_detail = cf_client.describe_stack_resource(
                        StackName=stack_name,
                        LogicalResourceId=r["LogicalResourceId"]
                    )
                    resource_props = resource_detail.get("StackResourceDetail", {})
                    if "ResourceStatusReason" in resource_props:
                        resource_info["ResourceStatusReason"] = resource_props["ResourceStatusReason"]
                except ClientError:
                    # If we can't get details, continue with what we have
                    pass
            
            formatted_resources.append(resource_info)
        
        # Get outputs
        outputs = {output["OutputKey"]: output["OutputValue"] for output in stack.get("Outputs", [])}
        
        # Categorize resources by status
        failed_resources = [
            r for r in formatted_resources
            if any(keyword in r["ResourceStatus"] for keyword in ["FAILED", "DELETE_FAILED", "CREATE_FAILED", "UPDATE_FAILED"])
        ]
        skipped_resources = [
            r for r in formatted_resources
            if "SKIP" in r["ResourceStatus"]
        ]
        
        return {
            "stack_info": {
                "StackName": stack["StackName"],
                "StackStatus": stack["StackStatus"],
                "CreationTime": stack["CreationTime"].isoformat() if "CreationTime" in stack else None,
                "Region": region,
            },
            "resources": formatted_resources,
            "resources_summary": {
                "total": len(formatted_resources),
                "failed": len(failed_resources),
                "skipped": len(skipped_resources),
                "healthy": len(formatted_resources) - len(failed_resources) - len(skipped_resources),
            },
            "failed_resources": failed_resources,
            "skipped_resources": skipped_resources,
            "outputs": outputs,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationError":
            raise ValueError(f"Bootstrap stack '{stack_name}' not found in region {region}") from e
        raise


def _empty_s3_bucket(bucket_name: str, region: str) -> Dict[str, Any]:
    """
    Empty an S3 bucket by deleting all objects and versions.
    
    Uses efficient batch deletion with pagination for large buckets.
    
    Args:
        bucket_name: Name of the S3 bucket (validated)
        region: AWS region
    
    Returns:
        Dictionary with operation status containing:
        - status: 'success', 'not_found', or 'error'
        - message: Human-readable message
        - bucket_name: Name of the bucket
        - objects_deleted: Number of objects deleted (if successful)
        - error: Error message (if failed)
    
    Raises:
        ValueError: If bucket_name is invalid
    """
    # Input validation
    if not bucket_name or not isinstance(bucket_name, str):
        raise ValueError("bucket_name must be a non-empty string")
    if not region or not isinstance(region, str):
        raise ValueError("region must be a non-empty string")
    
    # Validate bucket name format (basic check)
    if len(bucket_name) < 3 or len(bucket_name) > 63:
        raise ValueError(f"Invalid bucket name length: {bucket_name}")
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    s3_client = boto3.client("s3", region_name=region, config=config)
    s3_resource = boto3.resource("s3", region_name=region, config=config)
    
    try:
        logger.info(f"Emptying S3 bucket: {bucket_name}")
        bucket = s3_resource.Bucket(bucket_name)
        
        objects_deleted = 0
        versions_deleted = 0
        
        # Delete all object versions (more efficient for versioned buckets)
        try:
            version_paginator = s3_client.get_paginator("list_object_versions")
            for page in version_paginator.paginate(Bucket=bucket_name):
                versions = page.get("Versions", [])
                delete_markers = page.get("DeleteMarkers", [])
                
                # Batch delete versions
                if versions:
                    delete_keys = [
                        {"Key": v["Key"], "VersionId": v["VersionId"]}
                        for v in versions
                    ]
                    # Batch delete in chunks of 1000 (S3 limit)
                    for i in range(0, len(delete_keys), 1000):
                        chunk = delete_keys[i : i + 1000]
                        s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={"Objects": chunk, "Quiet": True},
                        )
                        versions_deleted += len(chunk)
                
                # Delete delete markers
                if delete_markers:
                    delete_keys = [
                        {"Key": dm["Key"], "VersionId": dm["VersionId"]}
                        for dm in delete_markers
                    ]
                    for i in range(0, len(delete_keys), 1000):
                        chunk = delete_keys[i : i + 1000]
                        s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={"Objects": chunk, "Quiet": True},
                        )
        except ClientError as e:
            # If versioning is not enabled, this is expected
            if e.response.get("Error", {}).get("Code") != "NoSuchBucket":
                logger.debug(f"Version listing failed (may not be versioned): {e}")
        
        # Delete all current objects (for non-versioned or remaining objects)
        try:
            object_paginator = s3_client.get_paginator("list_objects_v2")
            for page in object_paginator.paginate(Bucket=bucket_name):
                objects = page.get("Contents", [])
                if objects:
                    delete_keys = [{"Key": obj["Key"]} for obj in objects]
                    # Batch delete in chunks of 1000
                    for i in range(0, len(delete_keys), 1000):
                        chunk = delete_keys[i : i + 1000]
                        s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={"Objects": chunk, "Quiet": True},
                        )
                        objects_deleted += len(chunk)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "NoSuchBucket":
                logger.warning(f"Object listing failed: {e}")
        
        total_deleted = objects_deleted + versions_deleted
        logger.info(f"Successfully emptied S3 bucket: {bucket_name} ({total_deleted} items)")
        
        return {
            "status": "success",
            "message": f"S3 bucket '{bucket_name}' emptied successfully ({total_deleted} items)",
            "bucket_name": bucket_name,
            "objects_deleted": total_deleted,
        }
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        logger.error(f"Failed to empty S3 bucket {bucket_name}: {error_code} - {e}")
        
        if error_code == "NoSuchBucket":
            return {
                "status": "not_found",
                "message": f"S3 bucket '{bucket_name}' does not exist",
                "bucket_name": bucket_name,
            }
        return {
            "status": "error",
            "message": f"Failed to empty S3 bucket '{bucket_name}': {str(e)}",
            "bucket_name": bucket_name,
            "error": str(e),
            "error_code": error_code,
        }


def _empty_ecr_repository(repository_name: str, region: str) -> Dict[str, Any]:
    """
    Empty an ECR repository by deleting all images.
    
    Handles large repositories by batching deletions (ECR limit: 100 images per batch).
    
    Args:
        repository_name: Name of the ECR repository (validated)
        region: AWS region
    
    Returns:
        Dictionary with operation status containing:
        - status: 'success', 'not_found', or 'error'
        - message: Human-readable message
        - repository_name: Name of the repository
        - images_deleted: Number of images deleted (if successful)
        - error: Error message (if failed)
    
    Raises:
        ValueError: If repository_name is invalid
    """
    # Input validation
    if not repository_name or not isinstance(repository_name, str):
        raise ValueError("repository_name must be a non-empty string")
    if not region or not isinstance(region, str):
        raise ValueError("region must be a non-empty string")
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    ecr_client = boto3.client("ecr", region_name=region, config=config)
    
    try:
        logger.info(f"Emptying ECR repository: {repository_name}")
        
        # List all images in the repository
        paginator = ecr_client.get_paginator("list_images")
        image_ids = []
        
        for page in paginator.paginate(repositoryName=repository_name):
            image_ids.extend(page.get("imageIds", []))
        
        if not image_ids:
            logger.info(f"ECR repository '{repository_name}' is already empty")
            return {
                "status": "success",
                "message": f"ECR repository '{repository_name}' is already empty",
                "repository_name": repository_name,
                "images_deleted": 0,
            }
        
        # Delete images in batches (ECR limit: 100 per batch)
        total_deleted = 0
        batch_size = 100
        for i in range(0, len(image_ids), batch_size):
            batch = image_ids[i : i + batch_size]
            try:
                response = ecr_client.batch_delete_image(
                    repositoryName=repository_name,
                    imageIds=batch,
                )
                # Count successfully deleted images
                deleted = len(response.get("imageIds", []))
                total_deleted += deleted
                if deleted < len(batch):
                    failed = response.get("failures", [])
                    logger.warning(
                        f"Some images failed to delete in batch: {len(failed)} failures"
                    )
            except ClientError as e:
                logger.error(f"Failed to delete image batch: {e}")
                # Continue with next batch
                continue
        
        logger.info(
            f"Successfully deleted {total_deleted} image(s) from ECR repository: {repository_name}"
        )
        
        return {
            "status": "success",
            "message": f"Deleted {total_deleted} image(s) from ECR repository '{repository_name}'",
            "repository_name": repository_name,
            "images_deleted": total_deleted,
        }
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        logger.error(
            f"Failed to empty ECR repository {repository_name}: {error_code} - {e}"
        )
        
        if error_code == "RepositoryNotFoundException":
            return {
                "status": "not_found",
                "message": f"ECR repository '{repository_name}' does not exist",
                "repository_name": repository_name,
            }
        return {
            "status": "error",
            "message": f"Failed to empty ECR repository '{repository_name}': {str(e)}",
            "repository_name": repository_name,
            "error": str(e),
            "error_code": error_code,
        }


def _cleanup_resources_before_deletion(
    stack_name: str, region: str, dry_run: bool = False
) -> Dict[str, List[Dict[str, str]]]:
    """
    Clean up resources that need to be emptied before stack deletion.
    
    Args:
        stack_name: Name of the CloudFormation stack
        region: AWS region
        dry_run: If True, only report what would be cleaned without actually cleaning
    
    Returns:
        Dictionary with cleanup results
    """
    cf_client = boto3.client("cloudformation", region_name=region)
    
    cleanup_results = {
        "s3_buckets": [],
        "ecr_repositories": [],
        "errors": [],
    }
    
    try:
        # Get all stack resources
        paginator = cf_client.get_paginator("list_stack_resources")
        resources = []
        
        for page in paginator.paginate(StackName=stack_name):
            resources.extend(page.get("StackResourceSummaries", []))
        
        # Identify resources that need cleanup
        for resource in resources:
            resource_type = resource.get("ResourceType", "")
            logical_id = resource.get("LogicalResourceId", "")
            physical_id = resource.get("PhysicalResourceId", "")
            
            # Handle S3 buckets
            if resource_type == "AWS::S3::Bucket" and physical_id:
                if dry_run:
                    cleanup_results["s3_buckets"].append({
                        "logical_id": logical_id,
                        "bucket_name": physical_id,
                        "status": "would_empty",
                        "message": f"Would empty S3 bucket '{physical_id}'",
                    })
                else:
                    result = _empty_s3_bucket(physical_id, region)
                    result["logical_id"] = logical_id
                    cleanup_results["s3_buckets"].append(result)
            
            # Handle ECR repositories
            elif resource_type == "AWS::ECR::Repository" and physical_id:
                if dry_run:
                    cleanup_results["ecr_repositories"].append({
                        "logical_id": logical_id,
                        "repository_name": physical_id,
                        "status": "would_empty",
                        "message": f"Would empty ECR repository '{physical_id}'",
                    })
                else:
                    result = _empty_ecr_repository(physical_id, region)
                    result["logical_id"] = logical_id
                    cleanup_results["ecr_repositories"].append(result)
        
        return cleanup_results
    except ClientError as e:
        cleanup_results["errors"].append({
            "error": str(e),
            "message": f"Failed to list stack resources: {str(e)}",
        })
        return cleanup_results


def remove_bootstrap_stack(
    region: Optional[str] = None,
    qualifier: Optional[str] = None,
    wait: bool = False,
    cleanup_resources: bool = True,
) -> Dict[str, str]:
    """
    Remove the CDK bootstrap stack created by 'cdk bootstrap' command.
    
    This function automatically empties S3 buckets and ECR repositories before
    attempting to delete the stack, preventing deletion failures due to non-empty
    resources.
    
    Args:
        region: AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)
        qualifier: CDK bootstrap qualifier (defaults to None for default bootstrap).
                   Must match the qualifier used when bootstrapping. Use
                   `list_all_bootstrap_stacks()` to find available qualifiers.
        wait: Whether to wait for stack deletion to complete (default: False)
        cleanup_resources: Whether to automatically empty S3 buckets and ECR
                          repositories before deletion (default: True)
    
    Returns:
        Dictionary with deletion status information containing:
        - 'status': One of 'not_found', 'deletion_initiated', 'deletion_in_progress', 'deleted'
        - 'message': Human-readable status message
        - 'stack_name': Name of the stack being deleted
        - 'region': AWS region
        - 'cleanup_results': Results of resource cleanup (if cleanup_resources=True)
    
    Raises:
        RuntimeError: If deletion fails
    
    Examples:
        # Remove default bootstrap with automatic cleanup
        result = remove_bootstrap_stack(region="us-east-1", wait=True)
        
        # Remove bootstrap with qualifier
        result = remove_bootstrap_stack(region="us-east-1", qualifier="prod", wait=True)
        
        # Remove without automatic cleanup (manual cleanup required)
        result = remove_bootstrap_stack(region="us-east-1", cleanup_resources=False)
    """
    region = region or os.environ.get("CDK_DEFAULT_REGION", DEFAULT_REGION)
    stack_name = f"{STACK_NAME_PREFIX}{qualifier}" if qualifier else DEFAULT_STACK_NAME
    
    # Input validation
    if qualifier and not isinstance(qualifier, str):
        raise ValueError("qualifier must be a string or None")
    if not isinstance(region, str) or not region:
        raise ValueError("region must be a non-empty string")
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    cf_client = boto3.client("cloudformation", region_name=region, config=config)
    
    try:
        # Check if stack exists
        try:
            stack_response = cf_client.describe_stacks(StackName=stack_name)
            stack_status = stack_response["Stacks"][0]["StackStatus"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationError":
                return {
                    "status": "not_found",
                    "message": f"Bootstrap stack '{stack_name}' not found in region {region}",
                    "stack_name": stack_name,
                    "region": region,
                }
            raise
        
        # Clean up resources before deletion if requested
        cleanup_results = None
        if cleanup_resources:
            try:
                cleanup_results = _cleanup_resources_before_deletion(
                    stack_name=stack_name,
                    region=region,
                    dry_run=False,
                )
            except Exception as e:
                # Log cleanup errors but continue with deletion attempt
                cleanup_results = {
                    "errors": [{"error": str(e), "message": f"Cleanup failed: {str(e)}"}],
                    "s3_buckets": [],
                    "ecr_repositories": [],
                }
        
        # Delete the stack
        cf_client.delete_stack(StackName=stack_name)
        
        result = {
            "status": "deletion_initiated",
            "message": f"Bootstrap stack '{stack_name}' deletion initiated",
            "stack_name": stack_name,
            "region": region,
        }
        
        if cleanup_results:
            result["cleanup_results"] = cleanup_results
            cleanup_summary = []
            if cleanup_results.get("s3_buckets"):
                s3_count = len([r for r in cleanup_results["s3_buckets"] if r.get("status") == "success"])
                cleanup_summary.append(f"{s3_count} S3 bucket(s) emptied")
            if cleanup_results.get("ecr_repositories"):
                ecr_count = len([r for r in cleanup_results["ecr_repositories"] if r.get("status") == "success"])
                cleanup_summary.append(f"{ecr_count} ECR repository(ies) emptied")
            if cleanup_summary:
                result["message"] += f" (cleaned: {', '.join(cleanup_summary)})"
        
        # Wait for deletion if requested
        if wait:
            logger.info(f"Waiting for stack deletion to complete: {stack_name}")
            waiter = cf_client.get_waiter("stack_delete_complete")
            try:
                waiter.wait(
                    StackName=stack_name,
                    WaiterConfig={
                        "Delay": WAITER_DELAY,
                        "MaxAttempts": MAX_WAITER_ATTEMPTS,
                    },
                )
                result["status"] = "deleted"
                result["message"] = f"Bootstrap stack '{stack_name}' successfully deleted"
                if cleanup_results and cleanup_summary:
                    result["message"] += f" (cleaned: {', '.join(cleanup_summary)})"
                logger.info(f"Stack deletion completed: {stack_name}")
            except Exception as e:
                logger.warning(f"Error waiting for stack deletion: {e}")
                result["status"] = "deletion_in_progress"
                result["message"] = f"Deletion in progress. Error waiting: {str(e)}"
        
        return result
    except ClientError as e:
        raise RuntimeError(f"Failed to delete bootstrap stack: {e}") from e


def get_failed_resource_details(
    logical_resource_id: str,
    region: Optional[str] = None,
    qualifier: Optional[str] = None,
) -> Dict[str, str]:
    """
    Get detailed information about a failed or problematic resource.
    
    Args:
        logical_resource_id: The logical ID of the resource in the stack
        region: AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)
        qualifier: CDK bootstrap qualifier (defaults to None for default bootstrap)
    
    Returns:
        Dictionary with detailed resource information including:
        - ResourceStatus: Current status
        - ResourceStatusReason: Reason for failure (if available)
        - ResourceType: Type of resource
        - PhysicalResourceId: Physical resource ID
        - Metadata: Additional metadata
        - Recommendations: Suggested actions to resolve issues
    
    Raises:
        ValueError: If resource not found
        ClientError: If AWS API call fails
    """
    region = region or os.environ.get("CDK_DEFAULT_REGION", DEFAULT_REGION)
    stack_name = f"{STACK_NAME_PREFIX}{qualifier}" if qualifier else DEFAULT_STACK_NAME
    
    # Input validation
    if qualifier and not isinstance(qualifier, str):
        raise ValueError("qualifier must be a string or None")
    if not isinstance(region, str) or not region:
        raise ValueError("region must be a non-empty string")
    if not logical_resource_id or not isinstance(logical_resource_id, str):
        raise ValueError("logical_resource_id must be a non-empty string")
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    cf_client = boto3.client("cloudformation", region_name=region, config=config)
    
    try:
        resource_detail = cf_client.describe_stack_resource(
            StackName=stack_name,
            LogicalResourceId=logical_resource_id
        )
        
        resource = resource_detail.get("StackResourceDetail", {})
        status = resource.get("ResourceStatus", "")
        
        # Build recommendations based on status and resource type
        recommendations = []
        resource_type = resource.get("ResourceType", "")
        
        if "DELETE_FAILED" in status:
            if "ECR::Repository" in resource_type:
                recommendations.append(
                    "ECR repository may contain images. Delete all images first: "
                    "aws ecr list-images --repository-name <repo-name> --query 'imageIds[*]' --output json | "
                    "aws ecr batch-delete-image --repository-name <repo-name> --image-ids file:///dev/stdin"
                )
                recommendations.append(
                    "Or force delete the repository: "
                    "aws ecr delete-repository --repository-name <repo-name> --force"
                )
            elif "S3::Bucket" in resource_type:
                recommendations.append(
                    "S3 bucket may not be empty. Empty the bucket first or use: "
                    "aws s3 rb s3://<bucket-name> --force"
                )
            else:
                recommendations.append(
                    "Resource may have dependencies or protection enabled. "
                    "Check AWS Console for details and manually delete if needed."
                )
            recommendations.append(
                "After cleaning up the resource, retry stack deletion or continue deletion."
            )
        elif "CREATE_FAILED" in status or "UPDATE_FAILED" in status:
            recommendations.append(
                "Check the ResourceStatusReason for specific error details."
            )
            recommendations.append(
                "Review CloudFormation events for more context: "
                f"aws cloudformation describe-stack-events --stack-name {stack_name}"
            )
        
        return {
            "LogicalResourceId": resource.get("LogicalResourceId", logical_resource_id),
            "PhysicalResourceId": resource.get("PhysicalResourceId", "N/A"),
            "ResourceType": resource_type,
            "ResourceStatus": status,
            "ResourceStatusReason": resource.get("ResourceStatusReason", "No reason provided"),
            "LastUpdatedTimestamp": resource.get("LastUpdatedTimestamp", "").isoformat() if resource.get("LastUpdatedTimestamp") else None,
            "Metadata": resource.get("Metadata", {}),
            "Recommendations": recommendations,
        }
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationError":
            raise ValueError(
                f"Resource '{logical_resource_id}' not found in stack '{stack_name}' in region {region}"
            ) from e
        raise


def get_bootstrap_qualifiers(region: Optional[str] = None) -> List[str]:
    """
    Get a list of all qualifiers used for bootstrap stacks in the region.
    
    This is a convenience function that extracts qualifiers from bootstrap stack names.
    The default bootstrap (CDKToolkit) is not included in the returned list.
    
    Args:
        region: AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)
    
    Returns:
        List of qualifier strings (empty list if no qualified stacks exist)
    
    Examples:
        # Get all qualifiers
        qualifiers = get_bootstrap_qualifiers(region="us-east-1")
        for qualifier in qualifiers:
            print(f"Found qualifier: {qualifier}")
            resources = list_bootstrap_resources(qualifier=qualifier)
    """
    stacks = list_all_bootstrap_stacks(region=region)
    qualifiers = []
    
    for stack in stacks:
        stack_name = stack["StackName"]
        if stack_name != DEFAULT_STACK_NAME and stack_name.startswith(STACK_NAME_PREFIX):
            qualifier = stack_name.replace(STACK_NAME_PREFIX, "", 1)
            qualifiers.append(qualifier)
    
    return qualifiers


def list_all_bootstrap_stacks(region: Optional[str] = None) -> List[Dict[str, str]]:
    """
    List all CDK bootstrap stacks in the region.
    
    This function helps you discover all bootstrap stacks and their qualifiers.
    The default bootstrap stack is named 'CDKToolkit', while stacks with qualifiers
    are named 'CDKToolkit-<qualifier>'.
    
    Args:
        region: AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)
    
    Returns:
        List of bootstrap stacks with their information. Each stack dict contains:
        - 'StackName': Full stack name (e.g., 'CDKToolkit' or 'CDKToolkit-prod')
        - 'StackStatus': Current status of the stack
        - 'CreationTime': ISO format timestamp of creation
        - 'Region': AWS region
    
    Examples:
        # List all bootstrap stacks
        stacks = list_all_bootstrap_stacks(region="us-east-1")
        
        # Extract qualifiers from stack names
        for stack in stacks:
            name = stack['StackName']
            if name == 'CDKToolkit':
                print(f"Default bootstrap: {name}")
            else:
                qualifier = name.replace('CDKToolkit-', '')
                print(f"Qualified bootstrap: {name} (qualifier: {qualifier})")
    """
    region = region or os.environ.get("CDK_DEFAULT_REGION", DEFAULT_REGION)
    
    # Input validation
    if not isinstance(region, str) or not region:
        raise ValueError("region must be a non-empty string")
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    cf_client = boto3.client("cloudformation", region_name=region, config=config)
    
    try:
        # List all stacks and filter for CDKToolkit stacks
        paginator = cf_client.get_paginator("list_stacks")
        stacks = []
        
        for page in paginator.paginate(
            StackStatusFilter=[
                "CREATE_IN_PROGRESS",
                "CREATE_FAILED",
                "CREATE_COMPLETE",
                "ROLLBACK_IN_PROGRESS",
                "ROLLBACK_FAILED",
                "ROLLBACK_COMPLETE",
                "DELETE_IN_PROGRESS",
                "DELETE_FAILED",
                "UPDATE_IN_PROGRESS",
                "UPDATE_COMPLETE",
                "UPDATE_FAILED",
                "UPDATE_ROLLBACK_IN_PROGRESS",
                "UPDATE_ROLLBACK_FAILED",
                "UPDATE_ROLLBACK_COMPLETE",
                "REVIEW_IN_PROGRESS",
            ]
        ):
            for stack in page.get("StackSummaries", []):
                if stack["StackName"].startswith(DEFAULT_STACK_NAME):
                    stacks.append({
                        "StackName": stack["StackName"],
                        "StackStatus": stack["StackStatus"],
                        "CreationTime": stack.get("CreationTime", "").isoformat() if stack.get("CreationTime") else None,
                        "Region": region,
                    })
        
        return stacks
    except ClientError as e:
        raise RuntimeError(f"Failed to list bootstrap stacks: {e}") from e


def list_application_stacks(
    region: Optional[str] = None, environment: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    List all RAG Agent application stacks in the region.
    
    Application stacks are separate from bootstrap stacks and contain
    the actual application resources (API Gateway, Lambda, S3, DynamoDB, etc.).
    
    Args:
        region: AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)
        environment: Filter by environment (e.g., 'dev', 'prod'). If None, lists all.
    
    Returns:
        List of application stacks with their information. Each stack dict contains:
        - 'StackName': Full stack name (e.g., 'RAGAgentStack-dev')
        - 'StackStatus': Current status of the stack
        - 'CreationTime': ISO format timestamp of creation
        - 'Region': AWS region
        - 'Environment': Extracted environment from stack name
    """
    region = region or os.environ.get("CDK_DEFAULT_REGION", DEFAULT_REGION)
    
    # Input validation
    if not isinstance(region, str) or not region:
        raise ValueError("region must be a non-empty string")
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    cf_client = boto3.client("cloudformation", region_name=region, config=config)
    
    try:
        # List all stacks and filter for application stacks
        paginator = cf_client.get_paginator("list_stacks")
        stacks = []
        
        for page in paginator.paginate(
            StackStatusFilter=[
                "CREATE_IN_PROGRESS",
                "CREATE_FAILED",
                "CREATE_COMPLETE",
                "ROLLBACK_IN_PROGRESS",
                "ROLLBACK_FAILED",
                "ROLLBACK_COMPLETE",
                "DELETE_IN_PROGRESS",
                "DELETE_FAILED",
                "UPDATE_IN_PROGRESS",
                "UPDATE_COMPLETE",
                "UPDATE_FAILED",
                "UPDATE_ROLLBACK_IN_PROGRESS",
                "UPDATE_ROLLBACK_FAILED",
                "UPDATE_ROLLBACK_COMPLETE",
                "REVIEW_IN_PROGRESS",
            ]
        ):
            for stack in page.get("StackSummaries", []):
                stack_name = stack["StackName"]
                # Check if it's an application stack
                is_app_stack = any(
                    stack_name.startswith(prefix) for prefix in APP_STACK_PREFIXES
                )
                
                if is_app_stack:
                    # Extract environment from stack name
                    # Format: RAGAgentStack-{env} or RAGAgentLambdaStack-{env}
                    env = None
                    for prefix in APP_STACK_PREFIXES:
                        if stack_name.startswith(prefix):
                            env = stack_name.replace(prefix, "", 1)
                            break
                    
                    # Filter by environment if specified
                    if environment is None or env == environment:
                        stacks.append({
                            "StackName": stack_name,
                            "StackStatus": stack["StackStatus"],
                            "CreationTime": stack.get("CreationTime", "").isoformat()
                            if stack.get("CreationTime")
                            else None,
                            "Region": region,
                            "Environment": env,
                        })
        
        return stacks
    except ClientError as e:
        raise RuntimeError(f"Failed to list application stacks: {e}") from e


def remove_application_stacks(
    region: Optional[str] = None,
    environment: Optional[str] = None,
    wait: bool = False,
    cleanup_resources: bool = True,
) -> Dict[str, Any]:
    """
    Remove RAG Agent application stacks.
    
    This removes the application infrastructure stacks (RAGAgentStack and
    RAGAgentLambdaStack) which contain API Gateway, Lambda, S3 buckets, etc.
    
    Args:
        region: AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)
        environment: Environment to remove (e.g., 'dev', 'prod'). If None, removes all.
        wait: Whether to wait for stack deletion to complete (default: False)
        cleanup_resources: Whether to automatically empty S3 buckets and ECR
                          repositories before deletion (default: True)
    
    Returns:
        Dictionary with deletion results containing:
        - 'stacks_removed': List of stacks that were deleted
        - 'stacks_failed': List of stacks that failed to delete
        - 'total': Total number of stacks processed
        - 'region': AWS region
    """
    region = region or os.environ.get("CDK_DEFAULT_REGION", DEFAULT_REGION)
    
    # Input validation
    if not isinstance(region, str) or not region:
        raise ValueError("region must be a non-empty string")
    
    # List application stacks
    stacks = list_application_stacks(region=region, environment=environment)
    
    if not stacks:
        return {
            "stacks_removed": [],
            "stacks_failed": [],
            "total": 0,
            "region": region,
            "message": f"No application stacks found for environment '{environment or 'all'}' in region {region}",
        }
    
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        region_name=region,
    )
    cf_client = boto3.client("cloudformation", region_name=region, config=config)
    
    stacks_removed = []
    stacks_failed = []
    
    # Delete stacks in reverse order (Lambda stack first, then base stack)
    # to handle dependencies
    sorted_stacks = sorted(
        stacks, key=lambda s: s["StackName"], reverse=True
    )  # Lambda stack comes before base stack alphabetically
    
    for stack_info in sorted_stacks:
        stack_name = stack_info["StackName"]
        try:
            logger.info(f"Removing application stack: {stack_name}")
            
            # Clean up resources if requested
            if cleanup_resources:
                try:
                    _cleanup_resources_before_deletion(
                        stack_name=stack_name, region=region, dry_run=False
                    )
                except Exception as e:
                    logger.warning(f"Cleanup failed for {stack_name}: {e}")
                    # Continue with deletion anyway
            
            # Delete the stack
            cf_client.delete_stack(StackName=stack_name)
            
            if wait:
                logger.info(f"Waiting for stack deletion: {stack_name}")
                waiter = cf_client.get_waiter("stack_delete_complete")
                try:
                    waiter.wait(
                        StackName=stack_name,
                        WaiterConfig={
                            "Delay": WAITER_DELAY,
                            "MaxAttempts": MAX_WAITER_ATTEMPTS,
                        },
                    )
                    logger.info(f"Stack deleted successfully: {stack_name}")
                    stacks_removed.append(stack_name)
                except Exception as e:
                    logger.error(f"Error waiting for stack deletion {stack_name}: {e}")
                    stacks_failed.append({"stack": stack_name, "error": str(e)})
            else:
                stacks_removed.append(stack_name)
                
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.error(f"Failed to delete stack {stack_name}: {error_code} - {e}")
            stacks_failed.append({"stack": stack_name, "error": str(e), "error_code": error_code})
        except Exception as e:
            logger.error(f"Unexpected error deleting stack {stack_name}: {e}")
            stacks_failed.append({"stack": stack_name, "error": str(e)})
    
    return {
        "stacks_removed": stacks_removed,
        "stacks_failed": stacks_failed,
        "total": len(stacks),
        "region": region,
        "environment": environment,
        "message": f"Processed {len(stacks)} stack(s): {len(stacks_removed)} removed, {len(stacks_failed)} failed",
    }
