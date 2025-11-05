# AktuariBot — Chatbot Aktuaria (PSAK 219)

Ringkas: chatbot berbasis Flask dengan RAG yang mendukung dokumen sesi dan global, fokus PSAK 219. Dirancang dengan fallback defensif, logging, error handling, dan E2E tests.

## Setup (macOS Sequoia 15, python3)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # sesuaikan FLASK_PORT, LOG_LEVEL, dll.
```
Opsional: isi `OPENAI_API_KEY=sk-...` untuk mode LLM; isi `PINECONE_*` jika pakai Pinecone.

## Menjalankan
- Tanpa LLM (dev cepat):
```bash
FLASK_APP="app.app_factory:create_app" FLASK_ENV=development \
python3 -m flask run --port 5001
```
- Dengan LLM:
```bash
python3 app/main.py
```

## Alur E2E (Upload → AskProject → Jawaban)
1) Upload dokumen (opsional `session_id`, `project_id`):
```bash
curl -X POST http://localhost:5001/documents/upload \
  -F "files[]=@test_session_doc.md" -F "session_id=deep-session-xyz"
```
2) Bertanya via askproject (gabung sesi+global bila relevan):
```bash
curl -X POST http://localhost:5001/askproject \
  -H "Content-Type: application/json" \
  -d '{"question":"Ringkas kebijakan PSAK 219 proyek ini","project_id":"proj_abc","session_id":"deep-session-xyz"}'
```
3) Bertanya via askdeep (sesi-only, tanpa global):
```bash
curl -X POST http://localhost:5001/askdeep \
  -H "Content-Type: application/json" \
  -d '{"question":"Apa isi dokumen sesi?","session_id":"deep-session-xyz"}'
```

## Endpoint Utama
- `POST /askproject` — orkestrasi CoT, konteks proyek (sesi+global bila relevan)
- `POST /ask` — tanya umum, RAG moderat dan fallback
- `POST /askdeep` — RAG sesi-only (tanpa fallback global)
- `POST /documents/upload` — unggah dokumen sesi/global
- `GET /health` — health check

## Pengujian
```bash
pytest -q tests/test_e2e_askdeep.py
pytest -q tests/test_e2e_upload_and_askproject.py
pytest -q  # seluruh suite
```

## Fallback, Debugging, Error Handling, Logging
- Import LLM/Chroma dibungkus `try/except`; guard sebelum akses `self.llm`/vectorstore.
- Respons API gunakan `create_response(success, message, data)`; error ditangani rapi.
- Logging terpusat (`helpers.py`), tingkat dari `.env` (`LOG_LEVEL`).
- E2E tests disertakan untuk endpoint utama; tambah test saat menambah fitur.

## Deployment
- Docker:
```bash
docker build -t aktuaribot .
docker run --env-file .env -p 5001:5001 aktuaribot
```
- Makefile (umum): `make run`, `make test`, `make build`

## Dokumentasi Lanjutan
- `docs/ONBOARDING.md` — setup & run
- `docs/HANDOVER_BRIEF.md` — rencana kerja & DoD
- `docs/PROJECT_BRIEF.md` — arsitektur, flow E2E, API, metadata, fallback
- `docs/API_CONTEXT_GUIDE.md` — detail kontrak API & konteks
- `docs/ENHANCED_CONTEXT_INFORMATION.md` — penjelasan RAG & konteks
- `docs/MIGRATE_TO_PINECONE.md` — panduan migrasi backend vector
