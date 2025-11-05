"""Cache Manager for Semantic Retrieval Optimization.

This module implements caching mechanisms to improve retrieval performance
by storing frequently accessed query results and document embeddings.

Author: AI Assistant
Date: 2025-01-04
Version: 1.0.0
"""

import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from threading import Lock
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    access_count: int
    query_hash: str
    ttl: float = 300.0  # 5 minutes default TTL


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries to store
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Initialized LRU cache with max_size: {max_size}")
    
    def _generate_key(self, query: str, filters: Dict[str, Any], session_id: str) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            'query': query.lower().strip(),
            'filters': sorted(filters.items()) if filters else [],
            'session_id': session_id or 'none'
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, filters: Dict[str, Any], session_id: str) -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            query: Search query
            filters: Document filters
            session_id: Session identifier
            
        Returns:
            Cached data if found and valid, None otherwise
        """
        key = self._generate_key(query, filters, session_id)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                current_time = time.time()
                
                # Check if entry is still valid
                if current_time - entry.timestamp <= entry.ttl:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.access_count += 1
                    self.hits += 1
                    
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return entry.data
                else:
                    # Entry expired, remove it
                    del self.cache[key]
                    logger.debug(f"Cache entry expired for query: {query[:50]}...")
            
            self.misses += 1
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
    
    def put(self, query: str, filters: Dict[str, Any], session_id: str, data: Any, ttl: float = 300.0):
        """
        Store data in cache.
        
        Args:
            query: Search query
            filters: Document filters
            session_id: Session identifier
            data: Data to cache
            ttl: Time to live in seconds
        """
        key = self._generate_key(query, filters, session_id)
        
        with self.lock:
            # Remove oldest entries if cache is full
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.debug(f"Evicted oldest cache entry: {oldest_key}")
            
            # Add new entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                access_count=1,
                query_hash=key,
                ttl=ttl
            )
            
            self.cache[key] = entry
            logger.debug(f"Cached result for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': round(hit_rate, 2),
                'total_requests': total_requests
            }


class CacheManager:
    """Manages caching for semantic retrieval operations."""
    
    def __init__(self, max_cache_size: int = 1000, default_ttl: float = 300.0):
        """
        Initialize cache manager.
        
        Args:
            max_cache_size: Maximum number of cache entries
            default_ttl: Default time to live in seconds
        """
        self.query_cache = LRUCache(max_cache_size)
        self.embedding_cache = LRUCache(max_cache_size // 2)  # Smaller cache for embeddings
        self.default_ttl = default_ttl
        
        logger.info(f"Initialized CacheManager with max_size: {max_cache_size}, TTL: {default_ttl}s")
    
    def get_query_result(self, query: str, filters: Dict[str, Any], session_id: str) -> Optional[List[Tuple[Any, float]]]:
        """
        Get cached query result.
        
        Args:
            query: Search query
            filters: Document filters
            session_id: Session identifier
            
        Returns:
            Cached query results if found
        """
        return self.query_cache.get(query, filters, session_id)
    
    def cache_query_result(self, query: str, filters: Dict[str, Any], session_id: str, 
                          results: List[Tuple[Any, float]], ttl: Optional[float] = None):
        """
        Cache query result.
        
        Args:
            query: Search query
            filters: Document filters
            session_id: Session identifier
            results: Query results to cache
            ttl: Time to live (uses default if None)
        """
        cache_ttl = ttl or self.default_ttl
        self.query_cache.put(query, filters, session_id, results, cache_ttl)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding if found
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.embedding_cache.get(text_hash, {}, "embedding")
    
    def cache_embedding(self, text: str, embedding: List[float], ttl: Optional[float] = None):
        """
        Cache text embedding.
        
        Args:
            text: Original text
            embedding: Text embedding
            ttl: Time to live (uses default if None)
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_ttl = ttl or self.default_ttl * 2  # Embeddings last longer
        self.embedding_cache.put(text_hash, {}, "embedding", embedding, cache_ttl)
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        logger.info("All caches cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'query_cache': self.query_cache.get_stats(),
            'embedding_cache': self.embedding_cache.get_stats(),
            'total_memory_entries': len(self.query_cache.cache) + len(self.embedding_cache.cache)
        }
    
    def cleanup_expired_entries(self):
        """Remove expired entries from all caches."""
        current_time = time.time()
        
        # Clean query cache
        with self.query_cache.lock:
            expired_keys = []
            for key, entry in self.query_cache.cache.items():
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.query_cache.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired query cache entries")
        
        # Clean embedding cache
        with self.embedding_cache.lock:
            expired_keys = []
            for key, entry in self.embedding_cache.cache.items():
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.embedding_cache.cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired embedding cache entries")