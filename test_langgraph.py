"""Test script for LangGraph service.

This script tests the LangGraph service independently without starting the full Flask app.
"""

import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, '/home/why/Workspace/chatbot-aktuaria-new')

# Load environment variables
load_dotenv()

def test_imports():
    """Test if all required imports work."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        from app.services.langgraph_service import LangGraphRAGService
        print("✓ LangGraphRAGService imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import LangGraphRAGService: {e}")
        return False

def test_service_initialization():
    """Test if service can be initialized."""
    print("\n" + "=" * 60)
    print("Testing service initialization...")
    print("=" * 60)

    try:
        from app.services.langgraph_service import LangGraphRAGService

        # Check environment variables
        openai_key = os.getenv('OPENAI_API_KEY')
        pinecone_key = os.getenv('PINECONE_API_KEY')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'aktuaria-docs')

        print(f"\nEnvironment variables:")
        print(f"  OPENAI_API_KEY: {'Set ✓' if openai_key else 'NOT SET ✗'}")
        print(f"  PINECONE_API_KEY: {'Set ✓' if pinecone_key else 'NOT SET ✗'}")
        print(f"  PINECONE_INDEX_NAME: {index_name}")

        if not openai_key or not pinecone_key:
            print("\n✗ Missing required API keys. Please check your .env file.")
            return False

        print("\nInitializing LangGraph service...")
        service = LangGraphRAGService()
        print("✓ Service initialized successfully")

        return service
    except Exception as e:
        print(f"✗ Failed to initialize service: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_health_check(service):
    """Test service health check."""
    print("\n" + "=" * 60)
    print("Testing health check...")
    print("=" * 60)

    try:
        result = service.health_check()
        print(f"\nHealth check result:")
        print(f"  Status: {result.get('status')}")
        print(f"  Components:")
        for name, status in result.get('components', {}).items():
            print(f"    - {name}: {status}")
        print(f"  Index: {result.get('index_name')}")
        print(f"  Document count: {result.get('document_count', 0)}")

        if result.get('status') in ['healthy', 'degraded']:
            print("\n✓ Health check passed")
            return True
        else:
            print(f"\n✗ Health check failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query(service):
    """Test a simple query."""
    print("\n" + "=" * 60)
    print("Testing query processing...")
    print("=" * 60)

    test_question = "Apa itu PSAK 219?"
    print(f"\nQuestion: {test_question}")

    try:
        result = service.query(test_question)

        if result.get('success'):
            print(f"\n✓ Query processed successfully")
            print(f"\nAnswer:")
            print("-" * 60)
            print(result.get('answer', 'No answer'))
            print("-" * 60)
            return True
        else:
            print(f"\n✗ Query failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"✗ Query error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LangGraph Service Test Suite")
    print("=" * 60)

    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Stopping tests.")
        return

    # Test 2: Service initialization
    service = test_service_initialization()
    if not service:
        print("\n❌ Service initialization failed. Stopping tests.")
        print("\nNote: Make sure your Pinecone index exists and has documents.")
        return

    # Test 3: Health check
    if not test_health_check(service):
        print("\n⚠️  Health check failed, but continuing with query test...")

    # Test 4: Query
    test_query(service)

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)

if __name__ == '__main__':
    main()
