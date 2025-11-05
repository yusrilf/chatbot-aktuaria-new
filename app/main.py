"""Main application entry point for actuarial chatbot.

This module serves as the main entry point for the Flask application,
using the application factory pattern for better organization and testability.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import os
import logging

from app.app_factory import create_app, ensure_full_initialization
from app.config import config
from app.utils.helpers import validate_openai_key

logger = logging.getLogger(__name__)

# Create Flask application using factory pattern
app = create_app()

# Validate OpenAI API key on startup
if not validate_openai_key(config.OPENAI_API_KEY):
    logger.error("Invalid or missing OpenAI API key")
    raise ValueError("OpenAI API key is required")

# Ensure full initialization of services
ensure_full_initialization()



if __name__ == '__main__':
    # Port configuration with fallback chain
    # 1. Environment PORT variable (for cloud deployment)
    # 2. Config PORT setting
    # 3. Default 5000 (matches Dockerfile EXPOSE)
    port = int(os.environ.get("PORT", 
                              getattr(config, 'PORT', 5001)))
    
    logger.info(f"Starting Flask application on port {port}")
    app.run(debug=config.FLASK_DEBUG, host='0.0.0.0', port=port)