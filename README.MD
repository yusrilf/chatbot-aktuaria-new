# Actuarial Chatbot - Internal Knowledge Assistant

AI-powered chatbot untuk tim aktuaria internal dengan kemampuan pemrosesan dokumen markdown dan question-answering berbasis GPT-4o.

---

## Features

- 📚 Pemrosesan dokumen dari file markdown (.md)  
- 🤖 Conversational AI dengan GPT-4o  
- 🔍 Semantic search menggunakan OpenAI embeddings  
- 💾 Penyimpanan vektor dengan ChromaDB  
- 🧠 Memory percakapan dengan Langchain  
- 📊 Mendukung tabel dan formula matematika  
- 🚀 RESTful API menggunakan Flask  

---

## Quick Start

### 1. Setup Environment

```bash
# Clone atau download project
cd actuarial-chatbot

# Create virtual environment
python -m venv venv

# Aktifkan virtual environment
source venv/bin/activate  # Linux/Mac
# atau untuk Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env
```

* Edit file `.env` dan masukkan OpenAI API key Anda:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Run Application

```bash
# Jalankan Flask app
python app/main.py
atau
python3 -m app.app
```

* Server akan berjalan di `http://localhost:5001`

---

## API Endpoints

| Endpoint                | Method | Deskripsi                                            |
| ----------------------- | ------ | ---------------------------------------------------- |
| `/health`               | GET    | Health check                                         |
| `/input-docs`           | POST   | Upload dokumen (.md)                                 |
| `/ask`                  | POST   | Ajukan pertanyaan                                    |
| `/conversation/history` | GET    | Ambil riwayat percakapan (query param: `session_id`) |
| `/conversation/clear`   | POST   | Bersihkan memory percakapan                          |
| `/documents/stats`      | GET    | Statistik dokumen                                    |
| `/documents/search`     | POST   | Cari dalam dokumen                                   |
| `/documents/reset`      | POST   | Reset semua dokumen                                  |

---

## Contoh Penggunaan

### Upload Sample Documents

```bash
curl -X POST http://localhost:5001/input-docs \
  -F "files=@sample_docs/panduan_aktuaria.md" \
  -F "files=@sample_docs/laporan_keuangan.md" \
  -F "files=@sample_docs/rumus_aktuaria.md"
```

### Ask Questions

```bash
curl -X POST http://localhost:5001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Berapa dana pensiun yang dibutuhkan untuk 100 karyawan?",
    "session_id": "test_session"
  }'
```

---

## Project Structure

```
actuarial-chatbot/
├── app/                    # Main application
│   ├── main.py             # Flask app
│   ├── config.py           # Configuration
│   ├── models/             # Data models
│   ├── services/           # Business logic
│   └── utils/              # Helper functions
├── sample_docs/            # Sample markdown files
├── data/                   # Data storage
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## Customization

* **Menambah Jenis Dokumen Baru**
  Edit `app/services/document_processor.py` pada method `_extract_document_type()` untuk menambahkan logika pengenalan jenis dokumen baru.

* **Modifikasi Prompt AI**
  Edit `app/services/chat_service.py` pada method `_get_custom_prompt_template()` untuk menyesuaikan perilaku AI.

* **Mengatur Ukuran Chunk**
  Edit `app/config.py` untuk mengubah `CHUNK_SIZE` dan `CHUNK_OVERLAP` sesuai kebutuhan.

---

## Troubleshooting

**Masalah Umum:**

* **OpenAI API Error**: Pastikan API key valid dan memiliki kredit.
* **ChromaDB Error**: Hapus folder `data/vectorstore` dan restart aplikasi.
* **Memory Error**: Kurangi `CHUNK_SIZE` di config.
* **Import Error**: Pastikan dependencies sudah terinstall dengan benar.

**Logs:**
Cek file `app.log` untuk debugging.

---

## License

Internal use only - PT Asuransi Indonesia

