# See logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/rag-agent-dev" --query 'logGroups[*].logGroupName' --output text

# log tail
aws logs tail "/aws/lambda/rag-agent-dev-api" --since 10m --format short 2>/dev/null | tail -50

# Get Lambda config
aws lambda get-function-configuration --function-name rag-agent-dev-api --query 'Role' --output text


# Get IAM role info
aws iam get-role --role-name rag-agent-dev-app-role --query 'Role.AssumeRolePolicyDocument' --output json

# Get attached policies
aws iam list-attached-role-policies --role-name rag-agent-dev-app-role --output json && aws iam list-role-policies --role-name rag-agent-dev-app-role --output json

# Get policy details
aws iam get-role-policy --role-name rag-agent-dev-app-role --policy-name AppRoleDefaultPolicy1872B475 --query 'PolicyDocument.Statement' --output json

# Chat check
curl -s -X POST https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you help me with?"}'