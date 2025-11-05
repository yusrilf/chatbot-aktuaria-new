import os
import io
import logging
import pytest

from app.app_factory import create_app
from app.utils.singleton_manager import get_or_create_service, get_service_manager
from app.models.embeddings.vector_store_manager import VectorStoreManager

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """Minimal in-memory vector store for tests with filter support."""
    def __init__(self):
        self._docs = []

    def add_documents(self, docs):
        count_before = len(self._docs)
        self._docs.extend(docs or [])
        return len(self._docs) - count_before

    def similarity_search(self, query=" ", k=5, filter=None, **kwargs):
        docs = self._filter(filter)
        return docs[:k]

    def similarity_search_with_score(self, query=" ", k=5, filter=None, **kwargs):
        docs = self._filter(filter)
        return [(d, 1.0) for d in docs[:k]]

    def _filter(self, filter_dict):
        if not filter_dict:
            return list(self._docs)
        result = []
        for d in self._docs:
            meta = getattr(d, 'metadata', {}) or {}
            try:
                if all(meta.get(k) == v for k, v in filter_dict.items()):
                    result.append(d)
            except Exception:
                # Defensive: ignore bad metadata entries
                continue
        return result


@pytest.fixture(scope="module")
def test_app():
    """Create Flask app and patch vector store manager to use in-memory store."""
    app = create_app()
    app.testing = True

    # Initialize or get existing VectorStoreManager singleton
    vsm = get_or_create_service(VectorStoreManager, 'vector_store_manager')
    # Patch to in-memory vector store to avoid external dependencies
    vsm.chroma_client = None
    vsm.vectorstore = InMemoryVectorStore()
    # Ensure document manager points to patched vector store manager
    vsm.document_manager.vector_store_manager = vsm

    # Monkeypatch EnhancedActuarialChatService.ask_project to avoid LLM calls
    from app.services.enhanced_chat_service import EnhancedActuarialChatService

    async def _stub_ask_project(self, question: str, session_id: str):
        docs = vsm.document_manager.list_documents_for_session(session_id=session_id, include_global=True)
        return {
            'answer': f"stubbed answer for: {question}",
            'sources': [],
            'confidence': 0.5,
            'session_id': session_id,
            'mode': 'test_stub',
            'processing_time': 0.01,
            'available_docs_count': len(docs)
        }

    EnhancedActuarialChatService.ask_project = _stub_ask_project

    with app.test_client() as client:
        yield client


def _sample_json_path():
    """Resolve sample JSON file path used for E2E uploads."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidate = os.path.join(repo_root, 'sample_docs', 'file_rag_project_PT_abc.json')
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Sample JSON not found at {candidate}")
    return candidate


def test_upload_json_and_askproject_success(test_app):
    """E2E: upload JSON, ensure it’s processed and askproject returns stubbed answer."""
    session_id = "test-session-abc"

    # Upload JSON to /input-docs
    json_path = _sample_json_path()
    with open(json_path, 'rb') as fh:
        data = {
            'session_id': session_id,
            'files': [(fh, 'file_rag_project_PT_abc.json')]
        }
        resp = test_app.post('/input-docs', data=data, content_type='multipart/form-data')

    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.data}"
    payload = resp.get_json()
    assert payload['success'] is True
    assert payload['data']['json_files_processed'] == 1
    assert payload['data']['successful_files'] >= 1

    # Call /askproject
    q = {"question": "Apa isi utama dokumen proyek?", "session_id": session_id}
    resp2 = test_app.post('/askproject', json=q)
    assert resp2.status_code == 200
    payload2 = resp2.get_json()
    assert payload2['success'] is True
    data = payload2['data']
    assert data['mode'] == 'test_stub'
    assert data['session_id'] == session_id
    assert isinstance(data.get('available_docs_count'), int)
    assert data.get('answer', '').startswith('stubbed answer')


def test_reject_unsupported_file_extension(test_app):
    """E2E: upload unsupported file type returns error in results but overall endpoint responds."""
    session_id = "test-session-unsupported"
    fake_content = io.BytesIO(b"dummy")

    data = {
        'session_id': session_id,
        'files': [(fake_content, 'unsupported.exe')]
    }
    resp = test_app.post('/input-docs', data=data, content_type='multipart/form-data')

    assert resp.status_code == 200, f"Endpoint should respond: {resp.status_code}"
    payload = resp.get_json()
    assert payload['success'] is True  # Endpoint aggregates per-file results

    # Ensure the file is marked unsuccessful
    results = payload['data'].get('results', [])
    assert len(results) == 1
    r0 = results[0]
    assert r0['success'] is False
    assert 'Unsupported file type' in (r0.get('error') or '')