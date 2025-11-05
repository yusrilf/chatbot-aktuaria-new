"""Batch Processor for Optimized Document Processing.

This module provides batch processing capabilities to optimize
resource usage and improve performance for large document sets.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Callable, Iterator
from pathlib import Path
import time
from datetime import datetime
import math

from app.config import config
from app.utils.helpers import get_file_size
from .progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Optimized batch processor for document processing operations."""
    
    def __init__(self, 
                 default_batch_size: int = 10,
                 max_batch_size: int = 50,
                 min_batch_size: int = 1):
        """Initialize BatchProcessor.
        
        Args:
            default_batch_size: Default number of documents per batch
            max_batch_size: Maximum number of documents per batch
            min_batch_size: Minimum number of documents per batch
        """
        self.default_batch_size = default_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.progress_tracker = ProgressTracker()
        
        logger.info(f"BatchProcessor initialized with batch size {default_batch_size}")
    
    def create_batches(
        self, 
        file_paths: List[str], 
        batch_size: Optional[int] = None,
        optimize_by_size: bool = True
    ) -> List[List[str]]:
        """Create optimized batches from file paths.
        
        Args:
            file_paths: List of file paths to batch
            batch_size: Override default batch size
            optimize_by_size: Whether to optimize batches by file size
            
        Returns:
            List of batches, each containing file paths
        """
        if not file_paths:
            return []
        
        effective_batch_size = batch_size or self.default_batch_size
        effective_batch_size = max(self.min_batch_size, min(self.max_batch_size, effective_batch_size))
        
        if optimize_by_size:
            return self._create_size_optimized_batches(file_paths, effective_batch_size)
        else:
            return self._create_simple_batches(file_paths, effective_batch_size)
    
    def _create_simple_batches(self, file_paths: List[str], batch_size: int) -> List[List[str]]:
        """Create simple batches without size optimization.
        
        Args:
            file_paths: List of file paths to batch
            batch_size: Number of files per batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} simple batches from {len(file_paths)} files")
        return batches
    
    def _create_size_optimized_batches(self, file_paths: List[str], target_batch_size: int) -> List[List[str]]:
        """Create batches optimized by file size for balanced processing.
        
        Args:
            file_paths: List of file paths to batch
            target_batch_size: Target number of files per batch
            
        Returns:
            List of size-optimized batches
        """
        # Get file sizes
        file_info = []
        for file_path in file_paths:
            try:
                size = get_file_size(file_path)
                file_info.append((file_path, size))
            except Exception as e:
                logger.warning(f"Could not get size for {file_path}: {str(e)}")
                file_info.append((file_path, 0))
        
        # Sort by size (largest first for better distribution)
        file_info.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate target size per batch
        total_size = sum(info[1] for info in file_info)
        num_batches = max(1, math.ceil(len(file_paths) / target_batch_size))
        target_size_per_batch = total_size / num_batches
        
        # Create balanced batches
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for file_path, file_size in file_info:
            # Check if adding this file would exceed target size significantly
            if (current_batch and 
                len(current_batch) >= target_batch_size and
                current_batch_size + file_size > target_size_per_batch * 1.5):
                
                # Start new batch
                batches.append(current_batch)
                current_batch = [file_path]
                current_batch_size = file_size
            else:
                # Add to current batch
                current_batch.append(file_path)
                current_batch_size += file_size
        
        # Add remaining files
        if current_batch:
            batches.append(current_batch)
        
        logger.info(
            f"Created {len(batches)} size-optimized batches from {len(file_paths)} files "
            f"(avg {len(file_paths)/len(batches):.1f} files/batch)"
        )
        
        return batches
    
    def calculate_optimal_batch_size(
        self, 
        file_paths: List[str], 
        max_memory_mb: int = 512,
        target_processing_time: float = 30.0
    ) -> int:
        """Calculate optimal batch size based on file characteristics.
        
        Args:
            file_paths: List of file paths to analyze
            max_memory_mb: Maximum memory usage per batch in MB
            target_processing_time: Target processing time per batch in seconds
            
        Returns:
            Optimal batch size
        """
        if not file_paths:
            return self.default_batch_size
        
        # Sample files to estimate characteristics
        sample_size = min(10, len(file_paths))
        sample_files = file_paths[:sample_size]
        
        total_sample_size = 0
        for file_path in sample_files:
            try:
                size = get_file_size(file_path)
                total_sample_size += size
            except Exception:
                # Use default size estimate if file size cannot be determined
                total_sample_size += 1024 * 1024  # 1MB default
        
        # Calculate average file size
        avg_file_size_mb = (total_sample_size / sample_size) / (1024 * 1024)
        
        # Estimate batch size based on memory constraint
        memory_based_batch_size = max(1, int(max_memory_mb / (avg_file_size_mb * 2)))  # Factor of 2 for processing overhead
        
        # Estimate batch size based on processing time
        # Assume 1 second per MB of processing time (rough estimate)
        time_based_batch_size = max(1, int(target_processing_time / avg_file_size_mb))
        
        # Use the more conservative estimate
        optimal_batch_size = min(memory_based_batch_size, time_based_batch_size)
        
        # Apply bounds
        optimal_batch_size = max(self.min_batch_size, min(self.max_batch_size, optimal_batch_size))
        
        logger.info(
            f"Calculated optimal batch size: {optimal_batch_size} "
            f"(avg file size: {avg_file_size_mb:.2f}MB, "
            f"memory limit: {memory_based_batch_size}, "
            f"time limit: {time_based_batch_size})"
        )
        
        return optimal_batch_size
    
    def get_batch_statistics(self, batches: List[List[str]]) -> Dict[str, Any]:
        """Get statistics about created batches.
        
        Args:
            batches: List of batches to analyze
            
        Returns:
            Dictionary containing batch statistics
        """
        if not batches:
            return {
                'total_batches': 0,
                'total_files': 0,
                'avg_files_per_batch': 0,
                'min_files_per_batch': 0,
                'max_files_per_batch': 0,
                'batch_sizes': []
            }
        
        batch_sizes = [len(batch) for batch in batches]
        total_files = sum(batch_sizes)
        
        return {
            'total_batches': len(batches),
            'total_files': total_files,
            'avg_files_per_batch': total_files / len(batches),
            'min_files_per_batch': min(batch_sizes),
            'max_files_per_batch': max(batch_sizes),
            'batch_sizes': batch_sizes
        }
    
    def process_batches_sequentially(
        self,
        batches: List[List[str]],
        processor_func: Callable[[List[str]], Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Process batches sequentially with progress tracking.
        
        Args:
            batches: List of batches to process
            processor_func: Function to process each batch
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        total_files = sum(len(batch) for batch in batches)
        
        # Initialize progress tracking
        self.progress_tracker.reset(total_files, start_time)
        
        all_results = []
        batch_results = []
        
        logger.info(f"Processing {len(batches)} batches sequentially ({total_files} total files)")
        
        for i, batch in enumerate(batches):
            batch_start_time = time.time()
            
            try:
                logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")
                
                # Process the batch
                result = processor_func(batch)
                
                batch_processing_time = time.time() - batch_start_time
                
                # Track batch result
                batch_result = {
                    'batch_index': i,
                    'batch_size': len(batch),
                    'processing_time': batch_processing_time,
                    'success': result.get('success', False),
                    'files_processed': len(result.get('results', [])),
                    'error': result.get('error')
                }
                batch_results.append(batch_result)
                
                # Collect individual file results
                if result.get('success', False) and 'results' in result:
                    all_results.extend(result['results'])
                    
                    # Update progress for each file in the batch
                    for file_result in result['results']:
                        self.progress_tracker.update_progress(
                            file_result, progress_callback, 
                            is_error=not file_result.get('success', False)
                        )
                else:
                    # Handle batch failure
                    for file_path in batch:
                        error_result = {
                            'file_path': file_path,
                            'success': False,
                            'error': result.get('error', 'Batch processing failed'),
                            'processing_time': 0.0
                        }
                        all_results.append(error_result)
                        self.progress_tracker.update_progress(
                            error_result, progress_callback, is_error=True
                        )
                
                logger.info(
                    f"Batch {i+1} completed in {batch_processing_time:.2f}s "
                    f"({len(batch)} files)"
                )
                
            except Exception as e:
                logger.error(f"Error processing batch {i+1}: {str(e)}")
                
                # Handle batch exception
                batch_result = {
                    'batch_index': i,
                    'batch_size': len(batch),
                    'processing_time': time.time() - batch_start_time,
                    'success': False,
                    'files_processed': 0,
                    'error': str(e)
                }
                batch_results.append(batch_result)
                
                # Create error results for all files in failed batch
                for file_path in batch:
                    error_result = {
                        'file_path': file_path,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0.0
                    }
                    all_results.append(error_result)
                    self.progress_tracker.update_progress(
                        error_result, progress_callback, is_error=True
                    )
        
        total_time = time.time() - start_time
        
        # Calculate final statistics
        successful_batches = sum(1 for br in batch_results if br['success'])
        successful_files = sum(1 for r in all_results if r.get('success', False))
        
        final_result = {
            'success': True,
            'results': all_results,
            'batch_results': batch_results,
            'statistics': {
                'total_batches': len(batches),
                'successful_batches': successful_batches,
                'failed_batches': len(batches) - successful_batches,
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': total_files - successful_files,
                'total_time': total_time,
                'avg_time_per_batch': total_time / len(batches) if batches else 0,
                'avg_time_per_file': total_time / total_files if total_files > 0 else 0
            }
        }
        
        # Log final summary
        self.progress_tracker.log_final_summary()
        
        return final_result