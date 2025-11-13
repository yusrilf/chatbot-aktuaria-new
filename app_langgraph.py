"""Standalone Flask app for LangGraph endpoints only.

This minimal Flask app runs only the LangGraph service without depending on
other services that may have compatibility issues.

Run with: python app_langgraph.py
"""

import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Import and register LangGraph blueprint
from app.routes.langgraph_routes import langgraph_bp
app.register_blueprint(langgraph_bp)

# Root route
@app.route('/')
def root():
    """Root endpoint."""
    return {
        "name": "LangGraph RAG Service API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/langgraph/ask": "POST - Process question through RAG workflow",
            "/langgraph/health": "GET - Check service health",
            "/langgraph/info": "GET - Get service information"
        }
    }

# Health check
@app.route('/health')
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "langgraph-rag"
    }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))  # Different port to avoid conflict
    logger.info(f"Starting LangGraph Flask app on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
