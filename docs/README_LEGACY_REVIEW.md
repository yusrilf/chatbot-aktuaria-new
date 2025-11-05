# Review README Lama — Actuarial Chatbot (Advanced Knowledge Assistant)

Tujuan: mempertahankan README lama sebagai pengetahuan, sambil menandai bagian yang masih benar, perlu penyesuaian, atau tidak berlaku di kode saat ini. Tidak ada konten lama yang dihapus; dokumen ini berfungsi sebagai peta akurasi.

## Ringkasan Hasil

- Benar (valid sekarang)
  - Hybrid search (vector + lexical) tersedia via `app/models/hybrid_search.py`.
  - Adaptive chunking & metadata: `services/semantic_document_chunker.py`, `services/adaptive_chunking_strategy.py`, `services/chunk_metadata_extractor.py`.
  - Dukungan dokumen sesi & global: endpoint `POST /documents/upload`, preprocessor `global_docs_preprocessor.py`.
  - Health check: `GET /health`.
  - Docker & Makefile untuk build/run dasar.
  - Fallback defensif untuk LLM/Chroma; guard penggunaan berdasarkan ketersediaan dependency/env.
  - Logging terpusat via `app/utils/helpers.py` dan level via `.env` (`LOG_LEVEL`).
  - E2E tests untuk jalur utama: `tests/test_e2e_askdeep.py`, `tests/test_e2e_upload_and_askproject.py`.

- Perlu penyesuaian (akurat sebagian, perlu update)
  - Cara run Flask: gunakan `FLASK_APP="app.app_factory:create_app"` (bukan `app.main:app` atau `app.py`).
  - Versi Python: gunakan `python3` (macOS Sequoia 15, 3.10+/3.11), bukan hardcode 3.9.
  - Model LLM: "GPT-4.1" opsional; aplikasi berjalan tanpa LLM. Isi `OPENAI_API_KEY` jika dipakai.
  - Cohere rerank: opsional; aktif jika `COHERE_API_KEY` tersedia.
  - Port & path: konsistenkan penggunaan `5001` (README lama kadang menyebut `5000`).
  - Chroma DB path: cek `app/config.py` untuk `CHROMA_DB_PATH` default dan override via `.env`.

- Tidak berlaku/salah (tidak ditemukan di kode saat ini)
  - Endpoints berikut tidak ada atau bergeser kontrak:
    - `/input-docs`, `/inputglobaldocs`, `/upload-global-knowledge`, `/generate-global-docs-report`
    - `/documents/stats`, `/documents/search`, `/documents/reset`, `/documents/list`
    - `/datastory`, `/chat`, `/conversation/*`
    - `/search/*` (hybrid/vector/semantic sebagai public API)
    - `/api/storage/*` (refresh/clear/status) dan `/health/detailed`
  - Rujukan `app.py` untuk entry point (file ini tidak ada). Entry point: `app/main.py` atau factory.
  - Deployment Azure manual sebagai proses default pipeline (pipeline repo tidak mengikat pada Azure manual).

## Mapping Endpoint Lama → Kini

- `/input-docs` → `POST /documents/upload` (gunakan `session_id` untuk sesi; global via parameter/prosesor global).
- `/askproject` → tetap ada (orkestrasi CoT, konteks proyek/sesi+global bila relevan).
- `/ask` → tetap ada (pertanyaan umum, RAG moderat + fallback defensif).
- `/askdeep` → baru dipertegas sebagai sesi-only (tanpa fallback global).
- `/search/*` → tidak diekspos sebagai public endpoint; dikelola internal oleh service.
- `/api/storage/*` → tidak tersedia; gunakan proses manual/skrip jika perlu.

## Feature Lama — Status di Kode

- Hybrid Search: benar; weighting/komposisi spesifik perlu konfirmasi di `hybrid_search.py`.
- Cohere Rerank: opsional; aktif jika env diset. Tidak wajib.
- Adaptive Chunking: benar; ada strategi & ekstraksi metadata.
- Query Classification & Intent: terdapat `models/query_classifier.py` dan parser; validasi detail bergantung implementasi.
- Memory Management: konteks percakapan persisten tidak eksplisit; per-sesi konteks dokumen ada. History chat API tidak diekspos publik saat ini.
- Document Registry: penyimpanan dokumen & metadata diekstrak, namun tidak ada sistem registri formal tersendiri.
- Async Processing: ada modul `parallel_processing`, namun sebagian besar endpoint sinkron.
- Health Monitoring: `GET /health` tersedia.

## Menjalankan (Koreksi)

- Tanpa LLM (dev cepat):
```bash
FLASK_APP="app.app_factory:create_app" FLASK_ENV=development \
python3 -m flask run --port 5001
```
- Dengan LLM:
```bash
python3 app/main.py
```

## Keamanan, Logging, Error Handling

- Logging terpusat, level dari `.env` (`LOG_LEVEL`).
- Error handling: gunakan `create_response(success, message, data)` untuk konsistensi output.
- Fallback: bungkus import/akses dependency berat (OpenAI, Chroma) dalam `try/except` + guard.
- Secrets: simpan di `.env` dan jangan commit kunci.

## Pengujian

- E2E: jalankan `pytest -q tests/test_e2e_askdeep.py` dan `pytest -q tests/test_e2e_upload_and_askproject.py`.
- Tambah test saat menambah endpoint/fitur baru.

## Rekomendasi Dokumentasi

- Pertahankan README lama sebagai "Legacy Knowledge" (tidak dihapus).
- Gunakan `README.MD` ringkas untuk onboarding cepat, tautkan:
  - `docs/ONBOARDING.md`, `docs/HANDOVER_BRIEF.md`, `docs/PROJECT_BRIEF.md`.
- Tambahkan catatan di README lama: endpoint yang deprecated dan pengganti barunya.

## Next Steps (Opsional)

- Jika diinginkan: pulihkan README lama ke file terpisah `docs/README_LEGACY_FULL.md` dan tautkan dari README baru.
- Tambahkan diagram arsitektur ke `PROJECT_BRIEF.md` untuk memperjelas alur upload → RAG → jawaban.