"""Performance Monitoring Utility.

Utility untuk monitoring dan tracking performance aplikasi,
khususnya untuk Tree of Thought dan RAG operations.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from functools import wraps
from contextlib import contextmanager
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics untuk aplikasi."""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics = defaultdict(deque)
        self.active_timers = {}
        self.lock = threading.Lock()
        
    def start_timer(self, operation: str) -> str:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Timer ID for stopping the timer
        """
        timer_id = f"{operation}_{int(time.time() * 1000000)}"
        
        with self.lock:
            self.active_timers[timer_id] = {
                'operation': operation,
                'start_time': time.time(),
                'thread_id': threading.get_ident()
            }
        
        return timer_id
    
    def stop_timer(self, timer_id: str) -> Optional[float]:
        """
        Stop timing an operation.
        
        Args:
            timer_id: Timer ID from start_timer
            
        Returns:
            Elapsed time in seconds, or None if timer not found
        """
        with self.lock:
            if timer_id not in self.active_timers:
                logger.warning(f"Timer {timer_id} not found")
                return None
            
            timer_info = self.active_timers.pop(timer_id)
            elapsed_time = time.time() - timer_info['start_time']
            
            # Store metric
            operation = timer_info['operation']
            self.metrics[operation].append({
                'elapsed_time': elapsed_time,
                'timestamp': time.time(),
                'thread_id': timer_info['thread_id']
            })
            
            # Maintain max history
            if len(self.metrics[operation]) > self.max_history:
                self.metrics[operation].popleft()
            
            logger.info(f"Operation '{operation}' completed in {elapsed_time:.3f}s")
            return elapsed_time
    
    def end_timer(self, timer_id: str) -> Optional[float]:
        """
        Alias for stop_timer for backward compatibility.
        
        Args:
            timer_id: Timer ID from start_timer
            
        Returns:
            Elapsed time in seconds, or None if timer not found
        """
        return self.stop_timer(timer_id)
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all performance metrics.
        
        Returns:
            Dictionary containing all performance metrics
        """
        return self.get_all_stats()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary containing performance summary
        """
        stats = self.get_all_stats()
        
        if not stats:
            return {
                'total_operations': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'operations': {}
            }
        
        total_time = sum(stat.get('total_time', 0) for stat in stats.values() if stat)
        total_operations = sum(stat.get('count', 0) for stat in stats.values() if stat)
        average_time = total_time / total_operations if total_operations > 0 else 0.0
        
        return {
            'total_operations': total_operations,
            'total_time': total_time,
            'average_time': average_time,
            'operations': stats
        }
    
    @contextmanager
    def time_operation(self, operation: str):
        """
        Context manager for timing operations.
        
        Args:
            operation: Name of the operation
        """
        timer_id = self.start_timer(operation)
        try:
            yield
        finally:
            self.stop_timer(timer_id)
    
    def get_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Statistics dictionary
        """
        with self.lock:
            if operation not in self.metrics:
                return {}
            
            times = [m['elapsed_time'] for m in self.metrics[operation]]
            
            if not times:
                return {}
            
            return {
                'count': len(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time': sum(times),
                'recent_time': times[-1] if times else 0
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        with self.lock:
            # Copy metrics to avoid nested locking
            operations = list(self.metrics.keys())
            metrics_copy = {}
            for op in operations:
                if op in self.metrics:
                    times = [m['elapsed_time'] for m in self.metrics[op]]
                    if times:
                        metrics_copy[op] = {
                            'count': len(times),
                            'avg_time': sum(times) / len(times),
                            'min_time': min(times),
                            'max_time': max(times),
                            'total_time': sum(times),
                            'recent_time': times[-1]
                        }
                    else:
                        metrics_copy[op] = {}
            return metrics_copy
    
    def log_performance_summary(self):
        """Log performance summary for all operations."""
        stats = self.get_all_stats()
        
        if not stats:
            logger.info("No performance metrics available")
            return
        
        logger.info("=== PERFORMANCE SUMMARY ===")
        for operation, stat in stats.items():
            if stat:
                logger.info(f"{operation}: avg={stat['avg_time']:.3f}s, "
                           f"min={stat['min_time']:.3f}s, max={stat['max_time']:.3f}s, "
                           f"count={stat['count']}")
        logger.info("=== END PERFORMANCE SUMMARY ===")

def performance_timer(operation_name: str = None):
    """
    Decorator for timing function execution.
    
    Args:
        operation_name: Custom operation name, defaults to function name
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Get monitor from args/kwargs or create default
            monitor = None
            if hasattr(args[0], 'performance_monitor'):
                monitor = args[0].performance_monitor
            
            if monitor is None:
                # Use global monitor
                monitor = get_global_monitor()
            
            with monitor.time_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global performance monitor instance
_global_monitor = None
_monitor_lock = threading.Lock()

def get_global_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    
    if _global_monitor is None:
        with _monitor_lock:
            if _global_monitor is None:
                _global_monitor = PerformanceMonitor()
    
    return _global_monitor

def log_global_performance():
    """Log global performance summary."""
    monitor = get_global_monitor()
    monitor.log_performance_summary()