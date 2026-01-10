Available Endpoints
1. Health Check
curl https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/health

2. Query the RAG Agent
curl -X POST https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/query \  -H "Content-Type: application/json" \  -d '{    "query": "What is RAG?",    "include_sources": true  }'
3. Chat with the Agent
curl -X POST https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/chat \  -H "Content-Type: application/json" \  -d '{    "message": "Hello, how can you help me?"  }'
4. List Documents
curl https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/documents
5. Ingest a Document
curl -X POST https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/documents \  -H "Content-Type: application/json" \  -d '{    "content": "Your document content here...",    "title": "My Document",    "source": "manual"  }'
6. Get a Specific Document
curl https://9faay1yba3.execute-api.us-east-1.amazonaws.com/dev/documents/{document_id}
7. Delete a Document
curl -X DELETE https://9faay1yba3.execute-api.us-east-1.amazonaws.com
