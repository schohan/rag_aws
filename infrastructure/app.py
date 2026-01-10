#!/usr/bin/env python3
"""
CDK Application entry point.

Deploy with: cdk deploy --all
"""

import os

import aws_cdk as cdk

from stack import RAGAgentStack, RAGAgentLambdaStack


def main():
    """Create and configure the CDK app."""
    app = cdk.App()

    # Get environment from context or default to 'dev'
    environment = app.node.try_get_context("environment") or "dev"

    # Configure AWS environment
    env = cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    )

    # Create base infrastructure stack
    base_stack = RAGAgentStack(
        app,
        f"RAGAgentStack-{environment}",
        environment=environment,
        env=env,
        description=f"RAG Agent base infrastructure ({environment})",
    )

    # Create Lambda deployment stack (optional)
    lambda_stack = RAGAgentLambdaStack(
        app,
        f"RAGAgentLambdaStack-{environment}",
        base_stack=base_stack,
        environment=environment,
        env=env,
        description=f"RAG Agent Lambda deployment ({environment})",
    )

    # Add tags to all resources
    cdk.Tags.of(app).add("Project", "RAGAgent")
    cdk.Tags.of(app).add("Environment", environment)
    cdk.Tags.of(app).add("ManagedBy", "CDK")

    app.synth()


if __name__ == "__main__":
    main()

