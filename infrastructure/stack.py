"""
AWS CDK Stack for RAG Agent Infrastructure.

Creates all necessary AWS resources for the RAG application:
- S3 bucket for document storage
- S3 Vectors for vector embeddings
- DynamoDB table for metadata
- IAM roles and policies
- Bedrock Knowledge Base (optional)
"""

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

