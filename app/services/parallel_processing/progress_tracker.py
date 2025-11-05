"""Progress Tracker for Parallel Document Processing.

This module provides progress tracking capabilities for monitoring
the status of parallel document processing operations.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Thread-safe progress tracker for document processing operations."""
    
    def __init__(self):
        """Initialize ProgressTracker."""
        self._lock = threading.Lock()
        self.reset(0, time.time())
    
    def reset(self, total_files: int, start_time: float) -> None:
        """Reset progress tracking for a new batch.
        
        Args:
            total_files: Total number of files to process
            start_time: Start time of processing
        """
        with self._lock:
            self.total_files = total_files
            self.processed_files = 0
            self.successful_files = 0
            self.failed_files = 0
            self.start_time = start_time
            self.last_update_time = start_time
            self.total_chunks = 0
            self.processing_times = []
            
            logger.info(f"Progress tracker reset for {total_files} files")
    
    def update_progress(
        self, 
        result: Dict[str, Any], 
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        is_error: bool = False
    ) -> None:
        """Update progress with a processing result.
        
        Args:
            result: Processing result dictionary
            progress_callback: Optional callback function for progress updates
            is_error: Whether this update represents an error
        """
        with self._lock:
            self.processed_files += 1
            
            if is_error or not result.get('success', False):
                self.failed_files += 1
            else:
                self.successful_files += 1
                self.total_chunks += result.get('chunks_created', 0)
            
            # Track processing time
            processing_time = result.get('processing_time', 0.0)
            if processing_time > 0:
                self.processing_times.append(processing_time)
            
            self.last_update_time = time.time()
            
            # Generate progress info
            progress_info = self.get_progress_info()
            
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(progress_info)
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")
            
            # Log progress at intervals
            if self.processed_files % max(1, self.total_files // 10) == 0 or self.processed_files == self.total_files:
                logger.info(
                    f"Progress: {self.processed_files}/{self.total_files} "
                    f"({progress_info['percentage']:.1f}%) - "
                    f"Success: {self.successful_files}, Failed: {self.failed_files}"
                )
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information.
        
        Returns:
            Dictionary containing progress statistics
        """
        with self._lock:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Calculate percentage
            percentage = (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
            
            # Calculate rates
            files_per_second = self.processed_files / elapsed_time if elapsed_time > 0 else 0
            chunks_per_second = self.total_chunks / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate ETA
            remaining_files = self.total_files - self.processed_files
            eta_seconds = remaining_files / files_per_second if files_per_second > 0 else 0
            
            # Calculate average processing time
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times) 
                if self.processing_times else 0
            )
            
            return {
                'total_files': self.total_files,
                'processed_files': self.processed_files,
                'successful_files': self.successful_files,
                'failed_files': self.failed_files,
                'percentage': percentage,
                'elapsed_time': elapsed_time,
                'files_per_second': files_per_second,
                'chunks_per_second': chunks_per_second,
                'total_chunks': self.total_chunks,
                'eta_seconds': eta_seconds,
                'avg_processing_time': avg_processing_time,
                'success_rate': (self.successful_files / self.processed_files * 100) if self.processed_files > 0 else 0,
                'timestamp': datetime.now().isoformat(),
                'is_complete': self.processed_files >= self.total_files
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get final processing summary.
        
        Returns:
            Dictionary containing final processing statistics
        """
        progress_info = self.get_progress_info()
        
        with self._lock:
            # Additional summary statistics
            min_processing_time = min(self.processing_times) if self.processing_times else 0
            max_processing_time = max(self.processing_times) if self.processing_times else 0
            
            summary = {
                **progress_info,
                'min_processing_time': min_processing_time,
                'max_processing_time': max_processing_time,
                'total_processing_times': len(self.processing_times)
            }
            
            return summary
    
    def is_complete(self) -> bool:
        """Check if processing is complete.
        
        Returns:
            True if all files have been processed, False otherwise
        """
        with self._lock:
            return self.processed_files >= self.total_files
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            if not self.processing_times:
                return {
                    'avg_time': 0,
                    'min_time': 0,
                    'max_time': 0,
                    'total_samples': 0,
                    'throughput': 0
                }
            
            avg_time = sum(self.processing_times) / len(self.processing_times)
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
            elapsed = time.time() - self.start_time
            throughput = self.processed_files / elapsed if elapsed > 0 else 0
            
            return {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'total_samples': len(self.processing_times),
                'throughput': throughput,
                'total_elapsed': elapsed
            }
    
    def log_final_summary(self) -> None:
        """Log final processing summary."""
        summary = self.get_summary()
        
        logger.info(
            f"Processing Summary:\n"
            f"  Total Files: {summary['total_files']}\n"
            f"  Successful: {summary['successful_files']}\n"
            f"  Failed: {summary['failed_files']}\n"
            f"  Success Rate: {summary['success_rate']:.1f}%\n"
            f"  Total Time: {summary['elapsed_time']:.2f}s\n"
            f"  Average Time per File: {summary['avg_processing_time']:.2f}s\n"
            f"  Throughput: {summary['files_per_second']:.2f} files/s\n"
            f"  Total Chunks: {summary['total_chunks']}\n"
            f"  Chunks per Second: {summary['chunks_per_second']:.2f}"
        )