import os
import logging
import json
from typing import List, Dict, Any
from datetime import datetime

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key format"""
    return api_key and api_key.startswith('sk-') and len(api_key) > 20

def get_file_size(file_path: str) -> str:
    """Get human readable file size"""
    try:
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "Unknown"

def create_response(success: bool, message: str, data: Any = None) -> Dict[str, Any]:
    """Create standardized API response"""
    response = {
        'success': success,
        'message': message,
        'timestamp': datetime.now().isoformat()
    }
    
    if data is not None:
        response['data'] = data
    
    return response

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename.strip()

def format_number(number: float, decimal_places: int = 2) -> str:
    """Format number with Indonesian locale"""
    try:
        return f"{number:,.{decimal_places}f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return str(number)
    

def validate_files(files):
    return files and any(f.filename.strip() for f in files)
