import logging
import pytest

from app.app_factory import create_app
from app.utils.singleton_manager import get_or_create_service
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
                if all(meta.get(k) == v for k, v in (filter_dict or {}).items()):
                    result.append(d)
            except Exception:
                # Defensive: ignore bad metadata entries
                continue
        return result


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata or {}


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

    # Add sample documents for session and global
    session_id = "deep-session-xyz"
    vsm.vectorstore.add_documents([
        FakeDoc("konten sesi A", {"session_id": session_id, "filename": "session_A.md"}),
        FakeDoc("konten sesi B", {"session_id": session_id, "filename": "session_B.md"}),
        FakeDoc("konten global", {"session_id": "global", "filename": "global_doc.md"}),
    ])

    # Monkeypatch EnhancedActuarialChatService.ask_deep to avoid LLM calls
    from app.services.enhanced_chat_service import EnhancedActuarialChatService

    async def _stub_ask_deep(self, question: str, session_id_arg: str):
        results = vsm.similarity_search_with_score(
            query=question,
            session_id=session_id_arg,
            k=5,
            session_required=True,
            allow_fallback_to_global=False,
            return_placeholder_on_empty=False
        )
        sources = [{
            'filename': getattr(doc, 'metadata', {}).get('filename'),
            'session_id': getattr(doc, 'metadata', {}).get('session_id')
        } for doc, _ in results]
        return {
            'answer': f"stubbed deep for: {question}",
            'sources': sources,
            'confidence': 0.6 if sources else 0.3,
            'session_id': session_id_arg,
            'mode': 'test_stub_deep',
            'processing_time': 0.01
        }

    EnhancedActuarialChatService.ask_deep = _stub_ask_deep

    with app.test_client() as client:
        yield client


def test_askdeep_session_only_context(test_app):
    """E2E: ensure /askdeep returns only session-specific sources, excluding global."""
    session_id = "deep-session-xyz"
    q = {"question": "Apa isi dokumen sesi?", "session_id": session_id}
    resp = test_app.post('/askdeep', json=q)
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload['success'] is True
    data = payload['data']
    assert data['mode'] == 'test_stub_deep'
    assert data['session_id'] == session_id

    sources = data.get('sources', [])
    assert len(sources) >= 1
    # All sources must come from the requested session_id
    assert all(s.get('session_id') == session_id for s in sources)
    # Ensure no global source included
    assert not any(s.get('session_id') == 'global' for s in sources)


def test_askdeep_requires_question_field(test_app):
    """E2E: /askdeep should reject requests without 'question' field."""
    resp = test_app.post('/askdeep', json={"session_id": "deep-session-xyz"})
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload['success'] is False
    assert 'Question is required' in payload.get('message', '')