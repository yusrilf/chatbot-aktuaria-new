# PROJECT BRIEF — Chatbot Aktuaria (PSAK 219)

Ringkasan komprehensif seluruh proyek untuk memahami arsitektur, alur, komponen, dependensi, konfigurasi, cara menjalankan, pengujian, dan praktik pengembangan (fallback, debugging, error handling, logging, E2E tests).

## 1) Tujuan & Nilai
- Menjawab pertanyaan seputar PSAK 219 dengan konteks dokumen perusahaan dan materi global.
- Mendukung RAG per sesi pengguna, serta alur tanya jawab umum dan orkestrasi bertahap.

## 2) Arsitektur Tingkat Tinggi
- Framework: Flask dengan App Factory (`app/app_factory.py`) dan entry `app/main.py`.
- Komponen inti:
  - Routes (API): `app/routes/` — chat, dokumen, health.
  - Services: `app/services/` — chat, RAG/retrieval, document processing, parsers, calculators.
  - Models: `app/models/` — embedding dan pencarian (`embeddings/`, `search/`, `hybrid_search.py`).
  - Utils: `app/utils/` — logging & helpers, performance monitoring.
- Alur data: Upload/Ingest → Chunk & metadata → Embed → Index (Chroma/Pinecone) → Query → Retrieve & Rank → Compose jawaban.

## 3) Direktori & Modul Penting
- `app/routes/`
  - `chat_routes.py`: endpoint `/askproject`, `/ask`, `/askdeep`.
  - `document_routes.py`: upload & manajemen dokumen.
  - `api_routes.py`: penggabungan API dan util.
  - `health_routes.py`: health check.
- `app/services/`
  - `enhanced_chat_service.py`: EnhancedActuarialChatService (CoT, RAG ketat, fallback defensif).
  - `chat/chat_service.py`: ActuarialChatService modular (intent, kalkulasi, query), dengan fallback LLM.
  - Embeddings & Chroma opsional: `enhanced_embedding_service.py`, `enhanced_chroma_manager.py`.
  - Document processing: `document_processor.py`, `integrated_document_processor.py`, `global_docs_preprocessor.py`, chunking (`semantic_document_chunker.py`, `adaptive_chunking_strategy.py`, `chunk_metadata_extractor.py`).
  - Parsers & intent: `parsers/response_parser.py`, `chat/intent_classifier.py` (fallback heuristik ketika LLM tidak ada).
  - Kalkulator & domain: `services/calculators/`, `json_financial_processor*.py`.
- `app/models/`
  - `embeddings/`: `vector_store_manager.py` (facade), integrasi Chroma/Pinecone.
  - `search/`: `search_manager.py` (session/global filter, hybrid search), `hybrid_search.py`.
  - `query_classifier.py`: klasifikasi dasar query.
- `app/utils/`
  - `helpers.py`: logging setup, validasi API key, response standar.
  - `performance_monitor.py`: metrik performa internal.
  - `singleton_manager.py`: pola inisialisasi layanan.
- `scripts/`: batch embedding, setup integrasi, smoke test Pinecone, e2e health.
- `tests/`: E2E dan unit test termasuk `/askdeep` dan upload → askproject.
- `frontend/app.py`: contoh UI minimal.

## 4) Alur Chat & RAG
- EnhancedActuarialChatService (`ask_deep`):
  - RAG sesi-only: pencarian berdasarkan `session_id` tanpa fallback global.
  - Skor & sumber dicatat; bila dokumen sesi kosong, respons informatif (heuristik, tanpa global).
- ActuarialChatService (`ask`, `askproject`):
  - Menggunakan intent classification dan parser; dapat memakai dokumen global saat relevan.
- Retrieval & Vector Store:
  - `VectorStoreManager` → `SearchManager`.
  - Metode `similarity_search_with_score` dan `hybrid_similarity_search_with_score`.
  - Filter metadata berbasis `session_id`; fallback ke global dikendalikan oleh flag.

## 4A) Alur End-to-End (Upload → AskProject → Jawaban)
- 1. Upload Dokumen
  - Endpoint: `POST /documents/upload` (lihat `document_routes.py`).
  - Input: multipart `files[]`, opsional `session_id`, `project_id`.
  - Validasi: ukuran/tipe file, kebersihan nama (`helpers.py`).
  - Logging: jumlah file, `session_id`, `project_id`.
  - Output: daftar `doc_id` dan metadata dasar.
- 2. Pemrosesan Dokumen
  - Pipeline: `integrated_document_processor.py` → chunking (`semantic_document_chunker.py`, `adaptive_chunking_strategy.py`).
  - Ekstraksi metadata: `chunk_metadata_extractor.py` (isi `session_id`/`source=session|global`).
  - Fallback: jika LLM/embeddings tidak tersedia, gunakan heuristik sederhana, tetap hasilkan chunk.
- 3. Embedding & Indexing
  - Embedding: `enhanced_embedding_service.py` (OpenAIEmbeddings bila ada, fallback bila tidak).
  - Index: `vector_store_manager.py` ke backend `chroma`/`pinecone` sesuai `.env`.
  - Penandaan metadata ketat untuk mendukung filter retrieval.
- 4. Bertanya via AskProject
  - Endpoint: `POST /askproject` (lihat `chat_routes.py`).
  - Input: `{ "question": "...", "project_id": "...", "session_id": "optional" }`.
  - Orkestrasi: intent → penyusunan query → retrieval (boleh gabung sesi+global) → ranking.
  - Komposisi jawaban: parser (`response_parser.py`) dengan LLM bila tersedia, heuristik bila tidak.
  - Output: jawaban + sumber (doc_id, title, skor, session/global).
- 5. Error Handling & Logging
  - Gunakan `create_response(success, message, data)` untuk respons standar.
  - Tangkap error pada tiap tahap, log dengan level sesuai `LOG_LEVEL`.
  - Berikan pesan instruktif bila ketergantungan tidak tersedia (mis. sarankan unggah dokumen atau isi `.env`).
- 6. E2E Tests
  - Referensi: `tests/test_e2e_upload_and_askproject.py` untuk alur unggah → tanya.
  - Patch vector store in-memory saat test agar deterministik dan cepat.

## 4B) Alur AskDeep (Sesi-only)
- 1. Validasi Permintaan
  - Input wajib: `{ "question": "...", "session_id": "..." }`.
  - Jika tidak lengkap: `400` dengan pesan yang jelas.
- 2. Retrieval Sesi-only
  - Gunakan `VectorStoreManager.similarity_search_with_score(..., session_required=True, allow_fallback_to_global=False)`.
  - Bila dokumen sesi kosong: jawab heuristik yang informatif, TANPA global.
- 3. Komposisi Jawaban
  - Parser dan LLM opsional; heuristik bila LLM tidak ada.
  - Sertakan sumber hanya dari sesi yang sama.
- 4. Logging & E2E
  - Log jumlah dokumen, skor, dan filter yang diterapkan.
  - Uji via `tests/test_e2e_askdeep.py`.

## 5) Dependensi & Fallback
- LLM (OpenAI via `langchain_openai/ChatOpenAI`): impor dibungkus `try/except`. Jika tidak tersedia, sistem berjalan dengan heuristik (intent, parsing, jawaban dasar).
- Embeddings & Vector Store (Chroma/Pinecone): juga opsional, dengan guard sebelum akses.
- Prinsip: jangan biarkan `pytest`/server gagal karena modul eksternal tidak ada — gunakan fallback dan logging.

## 6) Konfigurasi & Lingkungan
- File: `app/config.py` memuat `.env`.
- Variabel penting:
  - `OPENAI_API_KEY`, `OPENAI_MODEL`.
  - `VECTOR_BACKEND` (`pinecone`/`chroma`), `CHROMA_DB_PATH`, `COLLECTION_NAME`.
  - Pinecone: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `PINECONE_ENV`.
  - Server: `FLASK_PORT`, `LOG_LEVEL`, `PERFORMANCE_LOGGING_ENABLED`.
- Gunakan `python3` di macOS Sequoia 15 (M2).

## 7) Menjalankan & Pengujian
- Tanpa LLM (disarankan untuk development):
  ```bash
  FLASK_APP="app.app_factory:create_app" FLASK_ENV=development \
  python3 -m flask run --port 5001
  ```
- Dengan LLM:
  ```bash
  python3 app/main.py
  ```
- Pengujian:
  - Fokus `/askdeep`: `pytest -q tests/test_e2e_askdeep.py`
  - Suite penuh: `pytest -q`
  - Skema test menambal VectorStore menjadi in-memory agar cepat dan deterministik.

## 8) Logging, Debugging, Error Handling
- Logging terpusat via `helpers.py`, tingkat `LOG_LEVEL` dari `.env`.
- Rute API mengembalikan `create_response(success, message, data)` standar.
- Error handlers di App Factory; hindari raise keras di jalur API.
- Performance monitor untuk metrik retrieval dan waktu respons.

## 9) Deployment & Operasional
- Docker:
  ```bash
  docker build -t aktuaribot .
  docker run --env-file .env -p 5001:5001 aktuaribot
  ```
- Makefile (contoh target umum):
  - `make run` → menjalankan factory Flask.
  - `make test` → menjalankan `pytest -q`.
  - `make build` → Docker build.

## 10) Data & Penyimpanan
- `data/chroma_db` untuk Chroma lokal.
- `uploaded_documents/` dan `data/documents/` untuk dokumen pengguna.
- `sample_docs/` berisi materi global PSAK 219 dan contoh sesi.

## 11) Frontend Smoke Test
- Jalankan contoh UI minimal:
  ```bash
  python3 frontend/app.py
  ```
- Uji koneksi API dengan payload sederhana ke `/ask` dan `/askdeep`.

## 12) Standar Pengembangan
- Selalu tambahkan fallback dan debugging untuk dependensi eksternal.
- Selalu sediakan error handling & logging di services dan routes.
- Selalu tambah E2E tests untuk endpoint baru.
- Jaga kerapian: refactor bila file >200 baris; fungsi kecil; penamaan jelas.

## 13) Keamanan & Secrets
- Simpan `OPENAI_API_KEY` dan `PINECONE_*` di `.env` (jangan commit).
- Validasi format OpenAI Key via helper; log peringatan jika invalid.
- Hindari menaruh PII mentah dalam metadata; sanitasi sebelum index.
- Batasi data sensitif di logs; gunakan level `INFO`/`WARN` sesuai kebutuhan.

## 14) Spesifikasi API (Kontrak)
- `POST /ask`
  - Body: `{ "question": "...", "session_id": "optional" }`
  - Respons sukses: `{ "success": true, "message": "ok", "data": { "answer": "...", "sources": [...] } }`
  - Error validasi: `{ "success": false, "message": "question wajib", "data": null }`
  - Status: `200`/`400`/`500`.
- `POST /askproject`
  - Body: `{ "question": "...", "project_id": "...", "session_id": "optional" }`
  - Respons mirip `/ask`, dapat orkestrasi CoT.
- `POST /askdeep`
  - Body: `{ "question": "...", "session_id": "..." }` (wajib dua-duanya)
  - Respons: sumber HANYA dari `session_id` yang diminta.
  - Error: `400` bila field hilang; tidak ada fallback ke global.
- `POST /documents/upload`
  - Multipart: `files[]`, opsional `session_id`, `project_id`.
  - Respons: daftar `doc_id`/metadata; error bila file invalid.
- `GET /health`
  - Respons: `{ "status": "ok", "uptime": ..., "version": ... }`.

## 15) Skema Metadata Vector Store
- Contoh metadata per chunk:
  ```json
  {
    "session_id": "deep-session-xyz",
    "source": "session", // atau "global"
    "doc_id": "doc_123",
    "chunk_id": 1,
    "title": "Pendahuluan PSAK 219",
    "project_id": "proj_abc",
    "created_at": "2025-08-20T04:40:37Z",
    "tags": ["psak219", "faq"]
  }
  ```
- Field wajib: `session_id` (untuk dokumen sesi), `source`, `doc_id`, `chunk_id`.
- Dampak filter: `/askdeep` memaksa `source=session` dan `session_id` sama; `/ask` dapat `source=global` tergantung flag.

## 16) Matriks Fallback
- LLM tersedia & valid → gunakan `ChatOpenAI`; bila tidak, pakai heuristik intent/parsing.
- VectorStore tersedia & ada dokumen sesi → gunakan retrieval; bila tidak:
  - `/askdeep`: tidak fallback ke global; berikan jawaban informatif/heuristik dan saran unggah dokumen.
  - `/ask`: boleh fallback ke global (tergantung konfigurasi).
- Keduanya tidak tersedia → respons instruktif, log `WARN`, dan sarankan setup `.env`/upload dokumen.

## 17) CI/CD Ringkas
- Workflow (`.github/workflows/main_aktuaribot-new.yml`):
  - Checkout → setup Python → install deps → jalankan `pytest` → build (opsional Docker) → artefak/notify.
  - Pastikan tes E2E kunci (`askdeep`) berjalan di pipeline.

## 18) Risiko & Mitigasi
- Modul eksternal hilang → fallback impor & guard; log peringatan.
- Filter metadata salah → tambah E2E skenario batas dan logging retrieval.
- Performa → mulai dengan `DEFAULT_K`, log jumlah dokumen & skor, optimasi bertahap.

## 19) Backlog & Roadmap
- Perkuat skenario batas `/askdeep` (sesi kosong, metadata rusak).
- Integrasi Pinecone produksi dan validasi indexing/filter.
- Tambah caching hasil retrieval; audit bundle dependensi untuk dev ringan.
- Dokumentasi API lebih rinci (lihat `docs/API_CONTEXT_GUIDE.md`).

## 20) Referensi Dokumen
- `docs/ONBOARDING.md` — setup & run.
- `docs/HANDOVER_BRIEF.md` — rencana kerja & DoD.
- `docs/API_CONTEXT_GUIDE.md`, `docs/ENHANCED_CONTEXT_INFORMATION.md` — detail konteks.
- `docs/MIGRATE_TO_PINECONE.md` — panduan migrasi.
- `README.MD` — ringkas proyek.