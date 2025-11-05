"""
Integrated Document Processor

This module provides a comprehensive document processing pipeline that integrates
the existing chunk metadata extractor with the new embedding and Chroma storage
capabilities. It handles the complete flow from document parsing to vector storage.

Author: AI Assistant
Date: 2025-01-27
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from app.services.enhanced_embedding_service import EnhancedEmbeddingService, EmbeddingResult, BatchEmbeddingStats
from app.services.enhanced_chroma_manager import EnhancedChromaManager, ChromaOperationResult
from app.services.chunk_metadata_extractor import ChunkMetadataExtractor
from app.services.global_docs_preprocessor import ChunkStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing operation."""
    success: bool
    document_name: str
    chunks_processed: int
    chunks_stored: int
    processing_time: float
    embedding_time: float
    storage_time: float
    error_message: Optional[str] = None
    metadata_summary: Optional[Dict[str, Any]] = None


@dataclass
class BatchProcessingStats:
    """Statistics for batch document processing."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks_processed: int
    total_chunks_stored: int
    total_processing_time: float
    average_time_per_document: float
    embedding_stats: Optional[BatchEmbeddingStats] = None
    storage_stats: Optional[List[ChromaOperationResult]] = None


class IntegratedDocumentProcessor:
    """Integrated document processor with embedding and storage capabilities."""
    
    def __init__(self, 
                 chroma_path: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 embedding_model: Optional[str] = None,
                 max_workers: int = 4):
        """Initialize the integrated document processor.
        
        Args:
            chroma_path: Path to Chroma database
            collection_name: Name of the collection
            embedding_model: Embedding model to use
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        
        # Initialize components
        self._initialize_components(chroma_path, collection_name, embedding_model)
        
        logger.info("Integrated Document Processor initialized successfully")
    
    def _initialize_components(self, 
                              chroma_path: Optional[str],
                              collection_name: Optional[str],
                              embedding_model: Optional[str]) -> None:
        """Initialize all processing components.
        
        Args:
            chroma_path: Path to Chroma database
            collection_name: Name of the collection
            embedding_model: Embedding model to use
        """
        try:
            # Initialize chunk metadata extractor
            self.metadata_extractor = ChunkMetadataExtractor()
            logger.info("Chunk metadata extractor initialized")
            
            # Initialize embedding service
            self.embedding_service = EnhancedEmbeddingService(
                chroma_path=chroma_path,
                collection_name=collection_name,
                embedding_model=embedding_model,
                max_workers=self.max_workers
            )
            logger.info("Enhanced embedding service initialized")
            
            # Initialize Chroma manager
            self.chroma_manager = EnhancedChromaManager(
                chroma_path=chroma_path,
                collection_name=collection_name,
                embedding_model=embedding_model
            )
            logger.info("Enhanced Chroma manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def process_single_document(self, 
                               file_path: str,
                               chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC,
                               store_in_chroma: bool = True) -> ProcessingResult:
        """Process a single document through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            chunk_strategy: Chunking strategy to use
            store_in_chroma: Whether to store results in Chroma
            
        Returns:
            ProcessingResult with processing details
        """
        start_time = time.time()
        doc_name = os.path.basename(file_path)
        
        try:
            logger.info(f"Starting processing of document: {doc_name}")
            
            # Step 1: Extract chunks with metadata
            extraction_start = time.time()
            chunks_with_metadata = self.metadata_extractor.extract_chunks_with_metadata(
                file_path, chunk_strategy
            )
            extraction_time = time.time() - extraction_start
            
            if not chunks_with_metadata:
                return ProcessingResult(
                    success=False,
                    document_name=doc_name,
                    chunks_processed=0,
                    chunks_stored=0,
                    processing_time=time.time() - start_time,
                    embedding_time=0,
                    storage_time=0,
                    error_message="No chunks extracted from document"
                )
            
            logger.info(f"Extracted {len(chunks_with_metadata)} chunks from {doc_name}")
            
            # Step 2: Convert to LangChain Documents
            documents = []
            for chunk_data in chunks_with_metadata:
                # Clean content (remove YAML frontmatter if present)
                content = self._clean_content(chunk_data['content'])
                
                # Create document with enhanced metadata
                doc = Document(
                    page_content=content,
                    metadata=chunk_data['metadata']
                )
                documents.append(doc)
            
            # Step 3: Generate embeddings and store (if requested)
            embedding_time = 0
            storage_time = 0
            chunks_stored = 0
            
            if store_in_chroma:
                # Generate embeddings and store
                embedding_start = time.time()
                
                # Use the embedding service for consistency
                embedding_result = self.embedding_service.process_single_document(
                    file_path, chunk_strategy
                )
                
                embedding_time = embedding_result.processing_time
                
                if embedding_result.success:
                    chunks_stored = embedding_result.chunk_count
                    logger.info(f"Successfully stored {chunks_stored} chunks for {doc_name}")
                else:
                    logger.error(f"Failed to store chunks for {doc_name}: {embedding_result.error_message}")
            
            # Step 4: Generate metadata summary
            metadata_summary = self._generate_metadata_summary(chunks_with_metadata)
            
            total_time = time.time() - start_time
            
            result = ProcessingResult(
                success=True,
                document_name=doc_name,
                chunks_processed=len(chunks_with_metadata),
                chunks_stored=chunks_stored,
                processing_time=total_time,
                embedding_time=embedding_time,
                storage_time=storage_time,
                metadata_summary=metadata_summary
            )
            
            logger.info(f"Successfully processed {doc_name}: {result.chunks_processed} chunks "
                       f"in {result.processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing document {doc_name}: {str(e)}"
            logger.error(error_msg)
            
            return ProcessingResult(
                success=False,
                document_name=doc_name,
                chunks_processed=0,
                chunks_stored=0,
                processing_time=time.time() - start_time,
                embedding_time=0,
                storage_time=0,
                error_message=error_msg
            )
    
    def process_documents_batch(self, 
                               file_paths: List[str],
                               chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC,
                               store_in_chroma: bool = True,
                               show_progress: bool = True) -> BatchProcessingStats:
        """Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            chunk_strategy: Chunking strategy to use
            store_in_chroma: Whether to store results in Chroma
            show_progress: Whether to show progress information
            
        Returns:
            BatchProcessingStats with processing statistics
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting batch processing of {len(file_paths)} documents")
            
            # Process documents individually for better error handling
            results = []
            
            for i, file_path in enumerate(file_paths, 1):
                if show_progress:
                    logger.info(f"Processing document {i}/{len(file_paths)}: {os.path.basename(file_path)}")
                
                result = self.process_single_document(
                    file_path, chunk_strategy, store_in_chroma
                )
                results.append(result)
                
                if show_progress:
                    status = "✅" if result.success else "❌"
                    logger.info(f"{status} {result.document_name}: {result.chunks_processed} chunks "
                               f"({result.processing_time:.2f}s)")
            
            # Calculate statistics
            total_time = time.time() - start_time
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            total_chunks_processed = sum(r.chunks_processed for r in successful)
            total_chunks_stored = sum(r.chunks_stored for r in successful)
            
            stats = BatchProcessingStats(
                total_documents=len(results),
                successful_documents=len(successful),
                failed_documents=len(failed),
                total_chunks_processed=total_chunks_processed,
                total_chunks_stored=total_chunks_stored,
                total_processing_time=total_time,
                average_time_per_document=total_time / len(results) if results else 0
            )
            
            logger.info(f"Batch processing completed: {stats.successful_documents}/{stats.total_documents} "
                       f"documents, {stats.total_chunks_processed} chunks processed, "
                       f"{stats.total_chunks_stored} chunks stored in {stats.total_processing_time:.2f}s")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise
    
    def process_directory(self, 
                         directory_path: str,
                         file_extensions: List[str] = ['.md', '.txt'],
                         chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC,
                         store_in_chroma: bool = True,
                         recursive: bool = True) -> BatchProcessingStats:
        """Process all documents in a directory.
        
        Args:
            directory_path: Path to the directory
            file_extensions: List of file extensions to process
            chunk_strategy: Chunking strategy to use
            store_in_chroma: Whether to store results in Chroma
            recursive: Whether to search recursively
            
        Returns:
            BatchProcessingStats with processing statistics
        """
        try:
            # Find all matching files
            file_paths = []
            directory = Path(directory_path)
            
            if recursive:
                for ext in file_extensions:
                    file_paths.extend(directory.rglob(f'*{ext}'))
            else:
                for ext in file_extensions:
                    file_paths.extend(directory.glob(f'*{ext}'))
            
            # Convert to strings
            file_paths = [str(path) for path in file_paths]
            
            logger.info(f"Found {len(file_paths)} files in {directory_path}")
            
            if not file_paths:
                return BatchProcessingStats(
                    total_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    total_chunks_processed=0,
                    total_chunks_stored=0,
                    total_processing_time=0,
                    average_time_per_document=0
                )
            
            # Process files in batch
            return self.process_documents_batch(
                file_paths, chunk_strategy, store_in_chroma, show_progress=True
            )
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise
    
    def upsert_document(self, 
                       file_path: str,
                       chunk_strategy: ChunkStrategy = ChunkStrategy.SEMANTIC) -> ProcessingResult:
        """Upsert a document (update if exists, insert if new).
        
        Args:
            file_path: Path to the document file
            chunk_strategy: Chunking strategy to use
            
        Returns:
            ProcessingResult with processing details
        """
        try:
            doc_name = os.path.basename(file_path)
            
            # Remove existing document from Chroma
            removal_result = self.chroma_manager.remove_documents_by_filename(doc_name)
            
            if removal_result.success and removal_result.affected_count > 0:
                logger.info(f"Removed {removal_result.affected_count} existing chunks for {doc_name}")
            
            # Process and store new version
            result = self.process_single_document(file_path, chunk_strategy, store_in_chroma=True)
            
            if result.success:
                logger.info(f"Successfully upserted document: {doc_name}")
            else:
                logger.error(f"Failed to upsert document: {doc_name}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error upserting document {file_path}: {str(e)}"
            logger.error(error_msg)
            
            return ProcessingResult(
                success=False,
                document_name=os.path.basename(file_path),
                chunks_processed=0,
                chunks_stored=0,
                processing_time=0,
                embedding_time=0,
                storage_time=0,
                error_message=error_msg
            )
    
    def _clean_content(self, content: str) -> str:
        """Clean content by removing YAML frontmatter and extra whitespace.
        
        Args:
            content: Raw content string
            
        Returns:
            Cleaned content string
        """
        try:
            lines = content.split('\n')
            
            # Remove YAML frontmatter if present
            if lines and lines[0].strip() == '---':
                # Find the end of frontmatter
                end_idx = 1
                while end_idx < len(lines) and lines[end_idx].strip() != '---':
                    end_idx += 1
                
                if end_idx < len(lines):
                    # Remove frontmatter lines
                    lines = lines[end_idx + 1:]
            
            # Join and clean whitespace
            cleaned = '\n'.join(lines).strip()
            
            # Remove excessive whitespace
            while '\n\n\n' in cleaned:
                cleaned = cleaned.replace('\n\n\n', '\n\n')
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error cleaning content: {str(e)}")
            return content.strip()
    
    def _generate_metadata_summary(self, chunks_with_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of metadata from processed chunks.
        
        Args:
            chunks_with_metadata: List of chunks with metadata
            
        Returns:
            Metadata summary dictionary
        """
        try:
            if not chunks_with_metadata:
                return {}
            
            # Extract common metadata
            first_chunk = chunks_with_metadata[0]['metadata']
            
            summary = {
                'document_info': {
                    'filename': first_chunk.get('filename', 'unknown'),
                    'doc_type': first_chunk.get('doc_type', 'unknown'),
                    'domain': first_chunk.get('domain', 'general'),
                    'version': first_chunk.get('version', '1.0'),
                    'last_updated': first_chunk.get('last_updated', '')
                },
                'chunk_stats': {
                    'total_chunks': len(chunks_with_metadata),
                    'avg_content_length': sum(len(chunk['content']) for chunk in chunks_with_metadata) / len(chunks_with_metadata),
                    'sections': len(set(chunk['metadata'].get('section_heading', '') for chunk in chunks_with_metadata))
                },
                'content_analysis': {
                    'keywords': first_chunk.get('keywords', []),
                    'difficulty': first_chunk.get('difficulty', 'basic'),
                    'scope': first_chunk.get('scope', 'general'),
                    'related_regulations': first_chunk.get('related_regulations', [])
                }
            }
            
            # Add numerical data if present
            numbers = first_chunk.get('numbers', {})
            if numbers:
                summary['numerical_data'] = numbers
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error generating metadata summary: {str(e)}")
            return {'error': str(e)}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            # Get Chroma stats
            chroma_stats = self.chroma_manager.get_document_stats()
            
            # Get health check
            health_check = self.chroma_manager.health_check()
            
            return {
                'processor_info': {
                    'max_workers': self.max_workers,
                    'components_initialized': True
                },
                'document_stats': {
                    'total_documents': chroma_stats.total_documents,
                    'total_chunks': chroma_stats.total_chunks,
                    'unique_documents': chroma_stats.unique_documents,
                    'doc_types': chroma_stats.doc_types,
                    'domains': chroma_stats.domains,
                    'difficulties': chroma_stats.difficulties,
                    'last_updated': chroma_stats.last_updated
                },
                'system_health': health_check,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}")
            return {
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def search_documents(self, 
                        query: str,
                        metadata_filter: Optional[Dict[str, Any]] = None,
                        k: int = 5) -> List[Tuple[Document, float]]:
        """Search documents using the integrated system.
        
        Args:
            query: Search query
            metadata_filter: Optional metadata filters
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            return self.chroma_manager.search_with_metadata_filter(
                query=query,
                metadata_filter=metadata_filter,
                k=k
            )
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []