# ONBOARDING — Chatbot Aktuaria (PSAK 219)

Tujuan: Membantu engineer baru memahami arsitektur, cara setup, menjalankan, menguji, dan mengembangkan fitur dengan prinsip fallback, debugging, error handling, logging, serta E2E tests. 

## Ringkas Arsitektur
- Backend Flask berbasis `app/app_factory.py` (blueprints, error handlers, root info).
- Layanan chat utama:
  - `app/services/enhanced_chat_service.py` — EnhancedActuarialChatService (CoT, RAG, fallback).
  - `app/services/chat/chat_service.py` — ActuarialChatService modular (intent, kalkulasi, query).
- Rute utama: `app/routes/chat_routes.py` → `/askproject`, `/ask`, `/askdeep`.
- Vector store facade: `app/models/embeddings/vector_store_manager.py` (delegasi ke SearchManager).
- Fallback dependency: import opsional untuk `langchain_openai`, `langchain_chroma`, `chromadb`, `OpenAIEmbeddings` dibungkus `try/except`.

## Prasyarat (macOS Sequoia 15)
- `python3` (disarankan 3.11+), `pip`, `venv`.
- Opsi LLM: `OPENAI_API_KEY` valid (format `sk-...` panjang > 20). Jika tidak ada, gunakan mode fallback.

## Setup Cepat
1. Buat virtualenv dan install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Siapkan `.env` (salin dari `.env.example` atau `.env.test`) dan isi minimal:
   - `FLASK_PORT=5001`
   - `LOG_LEVEL=INFO`
   - (Opsional) `OPENAI_API_KEY=sk-...`
   - (Opsional) Pinecone: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, dll.

## Menjalankan Aplikasi
- Tanpa LLM (disarankan untuk development cepat):
  ```bash
  FLASK_APP="app.app_factory:create_app" FLASK_ENV=development \
  python3 -m flask run --port 5001
  ```
- Dengan LLM (butuh `OPENAI_API_KEY` valid):
  ```bash
  python3 app/main.py
  ```
  Catatan: `app/main.py` melakukan validasi API key dan akan error jika tidak ada.

## Endpoint Utama
- `POST /askproject` — CoT orchestration, gunakan dokumen sesi dan global.
- `POST /ask` — pertanyaan umum, RAG moderat dan fallback.
- `POST /askdeep` — RAG sesi-only, tanpa fallback ke global.
  Contoh:
  ```bash
  curl -X POST http://localhost:5001/askdeep \
    -H "Content-Type: application/json" \
    -d '{"question":"Apa isi dokumen sesi?","session_id":"deep-session-xyz"}'
  ```

## Konfigurasi
- `app/config.py` memuat `.env` dengan `dotenv`.
- Kunci penting:
  - `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4.1`).
  - `VECTOR_BACKEND` (`pinecone`/`chroma`), `CHROMA_DB_PATH`, `COLLECTION_NAME`.
  - Pinecone: `PINECONE_*` (region, index, namespace, dimension, metric).
  - Logging: `LOG_LEVEL`, `PERFORMANCE_LOGGING_ENABLED`.
  - Port: `FLASK_PORT` (default 5001).

## Prinsip Pengembangan
- Fallback & debugging: bungkus import LLM/Chroma dengan `try/except`; guard sebelum akses `self.llm`/`self.vectorstore`.
- Error handling & logging: gunakan `logger` di semua operasi; pada rute API kembalikan `create_response(False, ...)` alih-alih raise.
- E2E tests wajib untuk endpoint baru (lihat bagian Pengujian).
- Kerapian kode: fungsi kecil, nama jelas; per file 100–200 baris (refactor bila melampaui).

## Pengujian
- Jalankan test khusus `/askdeep` (cepat dan isolasi):
  ```bash
  pytest -q tests/test_e2e_askdeep.py
  ```
  Test ini memaket `VectorStoreManager.vectorstore` menjadi in-memory store dan memastikan sumber hanya dari `session_id` diminta.
- Jalankan keseluruhan suite (berpotensi lama):
  ```bash
  pytest -q
  ```

## RAG & Vector Store
- `ask_deep` memakai:
  ```python
  VectorStoreManager.similarity_search_with_score(
    query=..., session_id=..., k=DEFAULT_K,
    session_required=True, allow_fallback_to_global=False,
    return_placeholder_on_empty=False
  )
  ```
- Filter metadata berbasis `session_id` ketat; sumber jawaban harus dari sesi yang sama.
- Pinecone produksi: isi `PINECONE_*` di `.env`, verifikasi indeks, dan pastikan filter bekerja.

## Troubleshooting
- `ModuleNotFoundError` (langchain_* / chroma): gunakan fallback atau `pip install langchain-openai langchain-chroma chromadb`.
- Server via `app/main.py` gagal: cek format `OPENAI_API_KEY`.
- Test lambat: jalankan file test spesifik dibanding seluruh suite.

## Struktur Direktori Singkat
- `app/` layanan, rute, config.
- `docs/` dokumentasi.
- `tests/` E2E/unit test.
- `data/` penyimpanan lokal.
- `sample_docs/` materi contoh (global/sesi).