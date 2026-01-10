"""
AWS Lambda handler for RAG Agent API.

Uses Mangum to wrap the FastAPI ASGI application for Lambda compatibility.
"""

import os
import sys

# Add the src directory to the path so we can import rag_agent
# This is needed when packaging the Lambda with the source code
src_path = os.path.join(os.path.dirname(__file__), "src")
if os.path.exists(src_path):
    sys.path.insert(0, src_path)

from mangum import Mangum
from rag_agent.api import app

# Create the Lambda handler
handler = Mangum(app, lifespan="off")

