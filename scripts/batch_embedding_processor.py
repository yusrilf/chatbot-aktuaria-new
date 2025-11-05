#!/usr/bin/env python3
"""
Batch Embedding Processor Script

This script demonstrates the complete pipeline for processing documents,
generating embeddings, and storing them in Chroma DB using the integrated
document processor system.

Usage:
    python scripts/batch_embedding_processor.py --input-dir data/sample_documents
    python scripts/batch_embedding_processor.py --single-file data/sample_documents/01_dasar_psak219.md
    python scripts/batch_embedding_processor.py --test-mode

Author: AI Assistant
Date: 2025-01-27
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.integrated_document_processor import IntegratedDocumentProcessor
from app.services.global_docs_preprocessor import ChunkStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_embedding.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_processor(chroma_path: str = "./vectorstore/chroma_db",
                   collection_name: str = "aktuaria_docs",
                   embedding_model: str = "text-embedding-3-large") -> IntegratedDocumentProcessor:
    """Setup the integrated document processor.
    
    Args:
        chroma_path: Path to Chroma database
        collection_name: Name of the collection
        embedding_model: Embedding model to use
        
    Returns:
        Configured IntegratedDocumentProcessor instance
    """
    try:
        logger.info("Initializing Integrated Document Processor...")
        
        processor = IntegratedDocumentProcessor(
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
            max_workers=4
        )
        
        logger.info("✅ Processor initialized successfully")
        return processor
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize processor: {str(e)}")
        raise


def process_single_document(processor: IntegratedDocumentProcessor,
                          file_path: str,
                          chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC) -> None:
    """Process a single document.
    
    Args:
        processor: The document processor instance
        file_path: Path to the document file
        chunk_strategy: Chunking strategy to use
    """
    try:
        logger.info(f"Processing single document: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return
        
        # Process the document
        result = processor.process_single_document(
            file_path=file_path,
            chunk_strategy=chunk_strategy,
            store_in_chroma=True
        )
        
        # Display results
        if result.success:
            logger.info(f"✅ Successfully processed: {result.document_name}")
            logger.info(f"   📄 Chunks processed: {result.chunks_processed}")
            logger.info(f"   💾 Chunks stored: {result.chunks_stored}")
            logger.info(f"   ⏱️  Processing time: {result.processing_time:.2f}s")
            logger.info(f"   🧠 Embedding time: {result.embedding_time:.2f}s")
            
            # Display metadata summary
            if result.metadata_summary:
                logger.info("   📊 Metadata Summary:")
                doc_info = result.metadata_summary.get('document_info', {})
                logger.info(f"      - Document Type: {doc_info.get('doc_type', 'unknown')}")
                logger.info(f"      - Domain: {doc_info.get('domain', 'general')}")
                logger.info(f"      - Version: {doc_info.get('version', '1.0')}")
                
                chunk_stats = result.metadata_summary.get('chunk_stats', {})
                logger.info(f"      - Total Sections: {chunk_stats.get('sections', 0)}")
                logger.info(f"      - Avg Content Length: {chunk_stats.get('avg_content_length', 0):.0f} chars")
        else:
            logger.error(f"❌ Failed to process: {result.document_name}")
            logger.error(f"   Error: {result.error_message}")
            
    except Exception as e:
        logger.error(f"❌ Error processing single document: {str(e)}")


def process_directory_batch(processor: IntegratedDocumentProcessor,
                           directory_path: str,
                           file_extensions: List[str] = ['.md', '.txt'],
                           chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC) -> None:
    """Process all documents in a directory.
    
    Args:
        processor: The document processor instance
        directory_path: Path to the directory
        file_extensions: List of file extensions to process
        chunk_strategy: Chunking strategy to use
    """
    try:
        logger.info(f"Processing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"❌ Directory not found: {directory_path}")
            return
        
        # Process the directory
        stats = processor.process_directory(
            directory_path=directory_path,
            file_extensions=file_extensions,
            chunk_strategy=chunk_strategy,
            store_in_chroma=True,
            recursive=True
        )
        
        # Display results
        logger.info("🎉 Batch processing completed!")
        logger.info(f"   📁 Total documents: {stats.total_documents}")
        logger.info(f"   ✅ Successful: {stats.successful_documents}")
        logger.info(f"   ❌ Failed: {stats.failed_documents}")
        logger.info(f"   📄 Total chunks processed: {stats.total_chunks_processed}")
        logger.info(f"   💾 Total chunks stored: {stats.total_chunks_stored}")
        logger.info(f"   ⏱️  Total processing time: {stats.total_processing_time:.2f}s")
        logger.info(f"   📊 Average time per document: {stats.average_time_per_document:.2f}s")
        
        # Calculate success rate
        if stats.total_documents > 0:
            success_rate = (stats.successful_documents / stats.total_documents) * 100
            logger.info(f"   📈 Success rate: {success_rate:.1f}%")
            
    except Exception as e:
        logger.error(f"❌ Error processing directory: {str(e)}")


def run_test_mode(processor: IntegratedDocumentProcessor) -> None:
    """Run test mode with sample documents.
    
    Args:
        processor: The document processor instance
    """
    try:
        logger.info("🧪 Running test mode...")
        
        # Test with sample documents directory
        sample_dir = "data/sample_documents"
        
        if os.path.exists(sample_dir):
            logger.info(f"Testing with sample documents from: {sample_dir}")
            process_directory_batch(processor, sample_dir)
        else:
            logger.warning(f"Sample directory not found: {sample_dir}")
            
            # Try to find any markdown files in the project
            project_root = Path(".")
            md_files = list(project_root.rglob("*.md"))
            
            if md_files:
                logger.info(f"Found {len(md_files)} markdown files in project")
                
                # Process first few files as test
                test_files = md_files[:3]  # Limit to 3 files for testing
                
                for file_path in test_files:
                    if "README" not in str(file_path):  # Skip README files
                        process_single_document(processor, str(file_path))
            else:
                logger.warning("No markdown files found for testing")
        
        # Display system stats
        display_system_stats(processor)
        
    except Exception as e:
        logger.error(f"❌ Error in test mode: {str(e)}")


def display_system_stats(processor: IntegratedDocumentProcessor) -> None:
    """Display comprehensive system statistics.
    
    Args:
        processor: The document processor instance
    """
    try:
        logger.info("📊 Getting system statistics...")
        
        stats = processor.get_processing_stats()
        
        logger.info("🔍 System Statistics:")
        logger.info(f"   📄 Total documents: {stats['document_stats']['total_documents']}")
        logger.info(f"   📝 Total chunks: {stats['document_stats']['total_chunks']}")
        logger.info(f"   🏷️  Document types: {', '.join(stats['document_stats']['doc_types'])}")
        logger.info(f"   🎯 Domains: {', '.join(stats['document_stats']['domains'])}")
        logger.info(f"   📊 Difficulties: {', '.join(stats['document_stats']['difficulties'])}")
        
        # Health check
        health = stats.get('system_health', {})
        if health.get('healthy', False):
            logger.info("   ✅ System health: OK")
        else:
            logger.warning("   ⚠️  System health: Issues detected")
            
    except Exception as e:
        logger.error(f"❌ Error getting system stats: {str(e)}")


def test_search_functionality(processor: IntegratedDocumentProcessor) -> None:
    """Test the search functionality.
    
    Args:
        processor: The document processor instance
    """
    try:
        logger.info("🔍 Testing search functionality...")
        
        # Test queries
        test_queries = [
            "PSAK 219",
            "aktuaria",
            "pensiun",
            "valuasi",
            "asuransi"
        ]
        
        for query in test_queries:
            logger.info(f"   Searching for: '{query}'")
            
            results = processor.search_documents(query=query, k=3)
            
            if results:
                logger.info(f"   Found {len(results)} results:")
                for i, (doc, score) in enumerate(results, 1):
                    metadata = doc.metadata
                    logger.info(f"      {i}. {metadata.get('filename', 'unknown')} "
                               f"(score: {score:.3f}) - {metadata.get('section_heading', 'No section')}")
            else:
                logger.info(f"   No results found for '{query}'")
                
    except Exception as e:
        logger.error(f"❌ Error testing search: {str(e)}")


def main():
    """Main function to run the batch embedding processor."""
    parser = argparse.ArgumentParser(description="Batch Embedding Processor for Aktuaria Chatbot")
    
    parser.add_argument("--input-dir", type=str, help="Directory containing documents to process")
    parser.add_argument("--single-file", type=str, help="Single file to process")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with sample documents")
    parser.add_argument("--chunk-strategy", type=str, default="semantic", 
                       choices=["semantic", "fixed", "paragraph"],
                       help="Chunking strategy to use")
    parser.add_argument("--chroma-path", type=str, default="./vectorstore/chroma_db",
                       help="Path to Chroma database")
    parser.add_argument("--collection-name", type=str, default="aktuaria_docs",
                       help="Name of the Chroma collection")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large",
                       help="OpenAI embedding model to use")
    parser.add_argument("--test-search", action="store_true", help="Test search functionality after processing")
    parser.add_argument("--show-stats", action="store_true", help="Show system statistics")
    
    args = parser.parse_args()
    
    try:
        # Convert chunk strategy string to enum
        chunk_strategy_map = {
            "semantic": ChunkStrategy.SEMANTIC,
            "fixed": ChunkStrategy.FIXED_SIZE,
            "paragraph": ChunkStrategy.PARAGRAPH
        }
        chunk_strategy = chunk_strategy_map.get(args.chunk_strategy, ChunkStrategy.SEMANTIC)
        
        # Setup processor
        processor = setup_processor(
            chroma_path=args.chroma_path,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model
        )
        
        # Process based on arguments
        if args.test_mode:
            run_test_mode(processor)
        elif args.single_file:
            process_single_document(processor, args.single_file, chunk_strategy)
        elif args.input_dir:
            process_directory_batch(processor, args.input_dir, chunk_strategy=chunk_strategy)
        else:
            logger.info("No processing mode specified. Use --help for options.")
            return
        
        # Optional: Test search functionality
        if args.test_search:
            test_search_functionality(processor)
        
        # Optional: Show system statistics
        if args.show_stats:
            display_system_stats(processor)
            
        logger.info("🎉 Batch embedding processing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()