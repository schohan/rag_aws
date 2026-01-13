#!/usr/bin/env python3
"""
CLI utility for managing CDK bootstrap resources.

Usage:
    python bootstrap_utils.py list [--region REGION] [--qualifier QUALIFIER]
    python bootstrap_utils.py remove [--region REGION] [--qualifier QUALIFIER] [--wait]
    python bootstrap_utils.py list-all [--region REGION]
    python bootstrap_utils.py list-qualifiers [--region REGION]
    python bootstrap_utils.py resource-details RESOURCE_ID [--region REGION] [--qualifier QUALIFIER]
    python bootstrap_utils.py list-app-stacks [--region REGION] [--environment ENV]
    python bootstrap_utils.py remove-app-stacks [--region REGION] [--environment ENV] [--wait]
"""

import argparse
import json
import sys
from typing import Optional

from stack import (
    get_bootstrap_qualifiers,
    get_failed_resource_details,
    list_all_bootstrap_stacks,
    list_application_stacks,
    list_bootstrap_resources,
    remove_application_stacks,
    remove_bootstrap_stack,
)


def format_output(data: dict, format_type: str = "pretty") -> str:
    """Format output as JSON or pretty print."""
    if format_type == "json":
        return json.dumps(data, indent=2, default=str)
    
    # Pretty print format
    output = []
    if "stack_info" in data:
        info = data["stack_info"]
        output.append("=" * 60)
        output.append("CDK Bootstrap Stack Information")
        output.append("=" * 60)
        output.append(f"Stack Name: {info.get('StackName', 'N/A')}")
        output.append(f"Status: {info.get('StackStatus', 'N/A')}")
        output.append(f"Region: {info.get('Region', 'N/A')}")
        if info.get("CreationTime"):
            output.append(f"Created: {info['CreationTime']}")
        output.append("")
    
    # Show summary if available
    if "resources_summary" in data:
        summary = data["resources_summary"]
        output.append("Resource Summary:")
        output.append("-" * 60)
        output.append(f"Total: {summary['total']}")
        output.append(f"Healthy: {summary['healthy']}")
        if summary['failed'] > 0:
            output.append(f"⚠️  Failed: {summary['failed']}")
        if summary['skipped'] > 0:
            output.append(f"⏭️  Skipped: {summary['skipped']}")
        output.append("")
    
    if "resources" in data:
        output.append("Resources:")
        output.append("-" * 60)
        
        # Show failed resources first if any
        if "failed_resources" in data and data["failed_resources"]:
            output.append("")
            output.append("❌ FAILED RESOURCES:")
            output.append("-" * 60)
            for i, resource in enumerate(data["failed_resources"], 1):
                output.append(f"{i}. {resource['LogicalResourceId']}")
                output.append(f"   Type: {resource['ResourceType']}")
                output.append(f"   Physical ID: {resource.get('PhysicalResourceId', 'N/A')}")
                output.append(f"   Status: {resource['ResourceStatus']}")
                if "ResourceStatusReason" in resource:
                    output.append(f"   Reason: {resource['ResourceStatusReason']}")
                output.append("")
            output.append("💡 Tip: Get detailed information about a failed resource:")
            output.append("   python bootstrap_utils.py resource-details <ResourceLogicalId>")
            output.append("")
        
        # Show skipped resources
        if "skipped_resources" in data and data["skipped_resources"]:
            output.append("")
            output.append("⏭️  SKIPPED RESOURCES:")
            output.append("-" * 60)
            for i, resource in enumerate(data["skipped_resources"], 1):
                output.append(f"{i}. {resource['LogicalResourceId']}")
                output.append(f"   Type: {resource['ResourceType']}")
                output.append(f"   Physical ID: {resource.get('PhysicalResourceId', 'N/A')}")
                output.append(f"   Status: {resource['ResourceStatus']}")
                if "ResourceStatusReason" in resource:
                    output.append(f"   Reason: {resource['ResourceStatusReason']}")
                output.append("")
        
        # Show all resources (or healthy ones if we've shown failed/skipped separately)
        show_all = not (data.get("failed_resources") or data.get("skipped_resources"))
        resources_to_show = data["resources"] if show_all else [
            r for r in data["resources"]
            if r not in (data.get("failed_resources", []) + data.get("skipped_resources", []))
        ]
        
        if resources_to_show:
            if not show_all:
                output.append("✅ HEALTHY RESOURCES:")
                output.append("-" * 60)
            for i, resource in enumerate(resources_to_show, 1):
                output.append(f"{i}. {resource['LogicalResourceId']}")
                output.append(f"   Type: {resource['ResourceType']}")
                output.append(f"   Physical ID: {resource.get('PhysicalResourceId', 'N/A')}")
                output.append(f"   Status: {resource['ResourceStatus']}")
                if "ResourceStatusReason" in resource:
                    output.append(f"   Reason: {resource['ResourceStatusReason']}")
                output.append("")
    
    if "outputs" in data and data["outputs"]:
        output.append("Stack Outputs:")
        output.append("-" * 60)
        for key, value in data["outputs"].items():
            output.append(f"{key}: {value}")
        output.append("")
    
    if "status" in data:
        output.append("=" * 60)
        output.append(f"Status: {data['status']}")
        output.append(f"Message: {data.get('message', 'N/A')}")
        if "stack_name" in data:
            output.append(f"Stack: {data['stack_name']}")
        if "region" in data:
            output.append(f"Region: {data['region']}")
        output.append("=" * 60)
    
    return "\n".join(output)


def cmd_list(args):
    """List bootstrap resources."""
    try:
        result = list_bootstrap_resources(
            region=args.region,
            qualifier=args.qualifier,
        )
        print(format_output(result, args.format))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error listing bootstrap resources: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_remove(args):
    """Remove bootstrap stack."""
    try:
        if not args.force:
            print(f"⚠️  This will remove the CDK bootstrap stack from your AWS account.")
            print(f"    You will need to run 'cdk bootstrap' again before deploying.")
            if args.cleanup:
                print(f"    Resources (S3 buckets, ECR repositories) will be automatically emptied.")
            response = input("Are you sure? [y/N] ")
            if response.lower() != "y":
                print("Cancelled.")
                sys.exit(0)
        
        result = remove_bootstrap_stack(
            region=args.region,
            qualifier=args.qualifier,
            wait=args.wait,
            cleanup_resources=args.cleanup,
        )
        
        # Format output with cleanup results
        if args.format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print(format_output(result, args.format))
            
            # Show detailed cleanup results
            if result.get("cleanup_results"):
                cleanup = result["cleanup_results"]
                print("")
                print("=" * 60)
                print("Resource Cleanup Results")
                print("=" * 60)
                
                if cleanup.get("s3_buckets"):
                    print("S3 Buckets:")
                    for bucket in cleanup["s3_buckets"]:
                        status_icon = "✅" if bucket.get("status") == "success" else "❌"
                        print(f"  {status_icon} {bucket.get('bucket_name', 'N/A')}: {bucket.get('message', 'N/A')}")
                    print("")
                
                if cleanup.get("ecr_repositories"):
                    print("ECR Repositories:")
                    for repo in cleanup["ecr_repositories"]:
                        status_icon = "✅" if repo.get("status") == "success" else "❌"
                        print(f"  {status_icon} {repo.get('repository_name', 'N/A')}: {repo.get('message', 'N/A')}")
                    print("")
                
                if cleanup.get("errors"):
                    print("Errors:")
                    for error in cleanup["errors"]:
                        print(f"  ❌ {error.get('message', 'Unknown error')}")
                    print("")
        
        if args.wait and result.get("status") == "deleted":
            print("✅ Bootstrap stack successfully deleted.")
    except Exception as e:
        print(f"Error removing bootstrap stack: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_all(args):
    """List all bootstrap stacks in the region."""
    try:
        stacks = list_all_bootstrap_stacks(region=args.region)
        
        if args.format == "json":
            print(json.dumps(stacks, indent=2, default=str))
        else:
            if not stacks:
                print(f"No CDK bootstrap stacks found in region {args.region or 'default'}.")
            else:
                print("=" * 60)
                print("CDK Bootstrap Stacks")
                print("=" * 60)
                for i, stack in enumerate(stacks, 1):
                    print(f"{i}. {stack['StackName']}")
                    print(f"   Status: {stack['StackStatus']}")
                    print(f"   Region: {stack['Region']}")
                    if stack.get("CreationTime"):
                        print(f"   Created: {stack['CreationTime']}")
                    # Show qualifier if present
                    if stack['StackName'] != 'CDKToolkit':
                        qualifier = stack['StackName'].replace('CDKToolkit-', '')
                        print(f"   Qualifier: {qualifier}")
                    print("")
    except Exception as e:
        print(f"Error listing bootstrap stacks: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_qualifiers(args):
    """List all qualifiers used for bootstrap stacks."""
    try:
        qualifiers = get_bootstrap_qualifiers(region=args.region)
        
        if args.format == "json":
            print(json.dumps({"qualifiers": qualifiers, "region": args.region or "default"}, indent=2))
        else:
            region_str = args.region or "default"
            if not qualifiers:
                print(f"No qualified bootstrap stacks found in region {region_str}.")
                print("Only the default bootstrap (CDKToolkit) exists.")
            else:
                print("=" * 60)
                print(f"Bootstrap Qualifiers (region: {region_str})")
                print("=" * 60)
                print(f"Found {len(qualifiers)} qualifier(s):")
                for i, qualifier in enumerate(qualifiers, 1):
                    print(f"  {i}. {qualifier}")
                    print(f"     Stack name: CDKToolkit-{qualifier}")
                print("")
                print("To list resources for a qualifier, use:")
                print(f"  python bootstrap_utils.py list --qualifier <qualifier>")
    except Exception as e:
        print(f"Error listing qualifiers: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_resource_details(args):
    """Get detailed information about a specific resource, especially failed ones."""
    try:
        details = get_failed_resource_details(
            logical_resource_id=args.resource_id,
            region=args.region,
            qualifier=args.qualifier,
        )
        
        if args.format == "json":
            print(json.dumps(details, indent=2, default=str))
        else:
            print("=" * 60)
            print(f"Resource Details: {details['LogicalResourceId']}")
            print("=" * 60)
            print(f"Type: {details['ResourceType']}")
            print(f"Status: {details['ResourceStatus']}")
            print(f"Physical ID: {details.get('PhysicalResourceId', 'N/A')}")
            if details.get('LastUpdatedTimestamp'):
                print(f"Last Updated: {details['LastUpdatedTimestamp']}")
            print("")
            
            if details.get('ResourceStatusReason'):
                print("Status Reason:")
                print("-" * 60)
                print(details['ResourceStatusReason'])
                print("")
            
            if details.get('Recommendations'):
                print("Recommendations:")
                print("-" * 60)
                for i, rec in enumerate(details['Recommendations'], 1):
                    print(f"{i}. {rec}")
                print("")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error getting resource details: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_app_stacks(args):
    """List all application stacks."""
    try:
        stacks = list_application_stacks(
            region=args.region,
            environment=args.environment,
        )
        
        if args.format == "json":
            print(json.dumps(stacks, indent=2, default=str))
        else:
            if not stacks:
                env_str = f" for environment '{args.environment}'" if args.environment else ""
                print(f"No application stacks found{env_str} in region {args.region or 'default'}.")
                print("")
                print("Note: Application stacks are separate from bootstrap stacks.")
                print("Bootstrap stacks are named 'CDKToolkit' and contain CDK deployment resources.")
                print("Application stacks are named 'RAGAgentStack-*' and 'RAGAgentLambdaStack-*'")
                print("  and contain your actual application resources (API Gateway, Lambda, etc.).")
            else:
                print("=" * 60)
                print("RAG Agent Application Stacks")
                print("=" * 60)
                print(f"Found {len(stacks)} stack(s):")
                print("")
                for i, stack in enumerate(stacks, 1):
                    print(f"{i}. {stack['StackName']}")
                    print(f"   Status: {stack['StackStatus']}")
                    print(f"   Environment: {stack.get('Environment', 'N/A')}")
                    print(f"   Region: {stack['Region']}")
                    if stack.get("CreationTime"):
                        print(f"   Created: {stack['CreationTime']}")
                    print("")
    except Exception as e:
        print(f"Error listing application stacks: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_remove_app_stacks(args):
    """Remove application stacks."""
    try:
        if not args.force:
            env_str = f" for environment '{args.environment}'" if args.environment else " (all environments)"
            print(f"⚠️  This will remove RAG Agent application stacks{env_str}.")
            print(f"    This includes API Gateway, Lambda functions, S3 buckets, DynamoDB tables, etc.")
            if args.cleanup:
                print(f"    Resources (S3 buckets, ECR repositories) will be automatically emptied.")
            response = input("Are you sure? [y/N] ")
            if response.lower() != "y":
                print("Cancelled.")
                sys.exit(0)
        
        result = remove_application_stacks(
            region=args.region,
            environment=args.environment,
            wait=args.wait,
            cleanup_resources=args.cleanup,
        )
        
        if args.format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            print("=" * 60)
            print("Application Stack Removal Results")
            print("=" * 60)
            print(result.get("message", "Operation completed"))
            print("")
            
            if result.get("stacks_removed"):
                print("✅ Successfully Removed:")
                for stack in result["stacks_removed"]:
                    print(f"  - {stack}")
                print("")
            
            if result.get("stacks_failed"):
                print("❌ Failed to Remove:")
                for failure in result["stacks_failed"]:
                    print(f"  - {failure.get('stack', 'Unknown')}: {failure.get('error', 'Unknown error')}")
                print("")
            
            print(f"Total: {result.get('total', 0)} stack(s) processed")
    except Exception as e:
        print(f"Error removing application stacks: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage CDK bootstrap resources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--region",
        type=str,
        help="AWS region (defaults to CDK_DEFAULT_REGION or us-east-1)",
    )
    parser.add_argument(
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List bootstrap resources")
    list_parser.add_argument(
        "--qualifier",
        type=str,
        help="CDK bootstrap qualifier (for custom bootstrap stacks)",
    )
    list_parser.set_defaults(func=cmd_list)
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove bootstrap stack")
    remove_parser.add_argument(
        "--qualifier",
        type=str,
        help="CDK bootstrap qualifier (for custom bootstrap stacks)",
    )
    remove_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for deletion to complete",
    )
    remove_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    remove_parser.add_argument(
        "--cleanup",
        action="store_true",
        default=True,
        help="Automatically empty S3 buckets and ECR repositories before deletion (default: True)",
    )
    remove_parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="Skip automatic cleanup of resources (manual cleanup required)",
    )
    remove_parser.set_defaults(func=cmd_remove)
    
    # List all command
    list_all_parser = subparsers.add_parser("list-all", help="List all bootstrap stacks in region")
    list_all_parser.set_defaults(func=cmd_list_all)
    
    # List qualifiers command
    qualifiers_parser = subparsers.add_parser(
        "list-qualifiers",
        help="List all qualifiers used for bootstrap stacks"
    )
    qualifiers_parser.set_defaults(func=cmd_list_qualifiers)
    
    # Resource details command
    details_parser = subparsers.add_parser(
        "resource-details",
        help="Get detailed information about a specific resource (useful for failed resources)"
    )
    details_parser.add_argument(
        "resource_id",
        type=str,
        help="Logical resource ID (e.g., ContainerAssetsRepository)",
    )
    details_parser.add_argument(
        "--qualifier",
        type=str,
        help="CDK bootstrap qualifier (for custom bootstrap stacks)",
    )
    details_parser.set_defaults(func=cmd_resource_details)
    
    # List application stacks command
    list_app_parser = subparsers.add_parser(
        "list-app-stacks",
        help="List all RAG Agent application stacks (separate from bootstrap)"
    )
    list_app_parser.add_argument(
        "--environment",
        type=str,
        help="Filter by environment (e.g., dev, prod)",
    )
    list_app_parser.set_defaults(func=cmd_list_app_stacks)
    
    # Remove application stacks command
    remove_app_parser = subparsers.add_parser(
        "remove-app-stacks",
        help="Remove RAG Agent application stacks (API Gateway, Lambda, etc.)"
    )
    remove_app_parser.add_argument(
        "--environment",
        type=str,
        help="Environment to remove (e.g., dev, prod). If not specified, removes all.",
    )
    remove_app_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for deletion to complete",
    )
    remove_app_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    remove_app_parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        default=True,
        help="Skip automatic cleanup of resources",
    )
    remove_app_parser.set_defaults(func=cmd_remove_app_stacks)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()

