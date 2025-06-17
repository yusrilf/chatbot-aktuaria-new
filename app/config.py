import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    OPENAI_MODEL = 'gpt-4o'
    EMBEDDING_MODEL = 'text-embedding-3-large'
    
    # ChromaDB Settings
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './data/vectorstore')
    COLLECTION_NAME = 'actuarial_documents'
    
    # Flask Settings
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Chat Settings
    MAX_CONTEXT_LENGTH = 4000
    SIMILARITY_THRESHOLD = 0.7
    TOP_K_RESULTS = 5
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

config = Config()

