#!/usr/bin/env python3
"""
Smoke test: Pinecone metadata slimming

This script exercises DocumentManager._slim_metadata_for_pinecone with a heavy
metadata payload to ensure the slimming logic trims to safe sizes and keeps
only allowed keys.
"""
import json
import random
import string
from typing import Any, Dict

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.documents import Document
from app.models.embeddings.document_manager import DocumentManager

class DummyVectorStoreManager:
    """Minimal stub to instantiate DocumentManager without full backend."""
    def __init__(self):
        self.vectorstore = None


def random_text(n: int) -> str:
    return ''.join(random.choice(string.ascii_letters + string.digits + ' ') for _ in range(n))


def build_heavy_metadata() -> Dict[str, Any]:
    return {
        # Allowed keys
        'filename': 'GLOBAL_GUIDE.md',
        'session_id': 'global',
        'is_global': True,
        'document_type': 'GENERAL_MARKDOWN',
        'doc_type': 'guide',
        'domain': 'actuarial',
        'scope': 'PSAK219',
        'version': '1.2.3',
        'last_updated': '2025-01-01',
        'chunk_id': 42,
        'section_heading': random_text(8000),  # excessive length
        'section_order': 7,
        'header_preview': random_text(6000),   # excessive length
        'chunk_size': 1200,
        'source': 'sample_docs/global_docs/GLOBAL_GUIDE.md',
        # Heavy keys that should be dropped
        'keywords': [random_text(200) for _ in range(300)],
        'entities': [{'type': 'ORG', 'text': random_text(300)} for _ in range(200)],
        'raw_json': {'a': random_text(12000)},
        'table_data': [[random_text(500) for _ in range(50)] for _ in range(20)],
        'debug_trace': random_text(10000),
    }


def main():
    dvm = DummyVectorStoreManager()
    dm = DocumentManager(vector_store_manager=dvm)

    meta = build_heavy_metadata()
    slim = dm._slim_metadata_for_pinecone(meta)

    # Compute approximate serialized size
    try:
        serialized = json.dumps(slim)
        size_bytes = len(serialized.encode('utf-8'))
    except Exception:
        size_bytes = -1

    print('Original keys:', len(meta))
    print('Slimmed keys:', len(slim))
    print('Approx serialized size (bytes):', size_bytes)
    print('Contains heavy keys (keywords/entities/raw_json/table_data/debug_trace):',
          any(k in slim for k in ['keywords','entities','raw_json','table_data','debug_trace']))

    # Show truncated lengths for long string fields
    for key in ['section_heading', 'header_preview']:
        val = slim.get(key, '')
        print(f"{key} length after trim:", len(val) if isinstance(val, str) else 'N/A')

    # Ensure essential fields exist
    essential = ['filename','session_id','is_global','doc_type','domain','source']
    missing = [k for k in essential if k not in slim]
    print('Missing essential keys:', missing)

    # Show a small preview
    print('Slim metadata preview:', json.dumps({k: (slim[k][:80] if isinstance(slim[k], str) else slim[k]) for k in list(slim.keys())[:8]}))

if __name__ == '__main__':
    main()