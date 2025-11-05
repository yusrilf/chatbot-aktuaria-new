# HANDOVER BRIEF — Chatbot Aktuaria

Dokumen singkat untuk onboarding cepat engineer baru agar langsung produktif, dengan ekspektasi, rencana, dan definisi selesai (DoD).

## Tujuan & Tanggung Jawab
- Memelihara dan mengembangkan chatbot aktuaria berfokus PSAK 219.
- Menjaga endpoint `/askdeep` selalu memakai konteks dokumen **sesi** (tanpa global).
- Menegakkan standar: fallback, debugging, error handling, logging, E2E test, refactor jika file >200 baris.

## Ekspektasi Teknis
- Dapat menjalankan server lokal di macOS Sequoia 15 dengan `python3`.
- Memahami alur RAG berbasis `session_id`, penggunaan `VectorStoreManager`, dan metadata filter.
- Menulis kode defensif (guard sebelum akses LLM/vectorstore) dan melengkapi E2E test.

## 90 Menit Pertama
1. Setup environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env  # sesuaikan FLASK_PORT, LOG_LEVEL
   ```
2. Jalankan tanpa LLM (development cepat):
   ```bash
   FLASK_APP="app.app_factory:create_app" FLASK_ENV=development \
   python3 -m flask run --port 5001
   ```
3. Verifikasi `/askdeep`:
   ```bash
   curl -X POST http://localhost:5001/askdeep \
     -H "Content-Type: application/json" \
     -d '{"question":"Apa isi dokumen sesi?","session_id":"deep-session-xyz"}'
   ```
4. Jalankan E2E fokus:
   ```bash
   pytest -q tests/test_e2e_askdeep.py
   ```

## Rencana 3–5 Hari
- Hari 1–2:
  - Pelajari `app/services/enhanced_chat_service.py::ask_deep` & `vector_store_manager.py` (metode similarity/hybrid).
  - Audit fallback/benteng impor (LLM/Chroma); pastikan test berjalan tanpa dependensi eksternal.
- Hari 3:
  - Tambah E2E scenario batas `/askdeep`: sesi tanpa dokumen, dokumen hanya global, metadata rusak.
  - Perkuat logging di retrieval (jumlah dokumen, skor, filter digunakan).
- Hari 4–5 (opsional produksi):
  - Integrasi Pinecone nyata — isi `PINECONE_*` di `.env`, verifikasi index dan filter `session_id`.
  - Update README dan docs bila ada perubahan arsitektur.

## Definisi Selesai (DoD)
- Server lokal berjalan tanpa LLM (factory) dan dengan LLM (entry `app/main.py`).
- `/askdeep` mengembalikan sumber HANYA dari `session_id` diminta, tidak ada global.
- E2E test untuk `/askdeep` lulus, termasuk skenario batas.
- Logging & error handling konsisten di service dan routes.
- Tidak ada `ModuleNotFoundError` saat pengujian berkat import opsional & guard.

## Risiko & Mitigasi
- Ketergantungan eksternal (OpenAI/Pinecone) → gunakan fallback saat dev/test.
- Performa retrieval → awali dengan `DEFAULT_K`, log jumlah dokumen, optimasi jika perlu.
- Test lama → jalankan file spesifik (`tests/test_e2e_askdeep.py`) sebelum suite penuh.

## Berkas Penting
- Layanan: `app/services/enhanced_chat_service.py`, `app/services/chat/chat_service.py`.
- Rute: `app/routes/chat_routes.py`.
- Vector store: `app/models/embeddings/vector_store_manager.py`, `app/models/embeddings/search_manager.py`.
- Konfigurasi: `app/config.py`, `.env.example`, `.env.test`.
- Test: `tests/test_e2e_askdeep.py`.

## Standar Koding
- Fallback & guard sebelum akses LLM/vectorstore; jangan raise keras di jalur API.
- Tambahkan E2E untuk endpoint baru dan skenario batas.
- Per file target 100–200 baris, refactor bila melewati.
- Logging di level yang sesuai; gunakan `create_response()` untuk respons API standar.

## Checklist Serah Terima
- [ ] Bisa start server lokal tanpa LLM (factory) dan dengan LLM (main).
- [ ] `/askdeep` hanya sumber sesi, tidak global.
- [ ] `pytest -q tests/test_e2e_askdeep.py` lulus.
- [ ] Fallback & logging konsisten; tidak ada error keras saat test.
- [ ] Dokumentasi ini dibaca dan dipahami.