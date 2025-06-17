Berikut adalah versi **Readme** yang telah **diperbaiki, dirapikan, dan ditambahkan endpoint `/ask` dan `/askproject`** dengan penjelasan parameter yang dibutuhkan. Saya juga memperbaiki tata bahasa dan menyelaraskan gaya penulisan:

---

# Actuarial Chatbot - Internal Knowledge Assistant

AI-powered chatbot untuk tim aktuaria internal, dengan kemampuan memproses dokumen Markdown dan menjawab pertanyaan berbasis GPT-4o.

---

## ✨ Features

* 📚 Pemrosesan dokumen Markdown (.md)
* 🤖 Conversational AI dengan GPT-4o
* 🔍 Semantic search menggunakan OpenAI Embeddings
* 💾 Penyimpanan vektor menggunakan ChromaDB
* 🧠 Memory percakapan via Langchain
* 📊 Mendukung tabel dan formula matematika
* 🚀 RESTful API berbasis Flask

---

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Clone repository
cd actuarial-chatbot

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
source venv/bin/activate  # Untuk Linux/Mac
# atau untuk Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi Environment

```bash
# Salin file template environment
cp .env.example .env
```

Edit `.env` dan isi dengan API key OpenAI Anda:

```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Jalankan Aplikasi

```bash
# Menjalankan Flask app
python app/main.py
# atau
python3 -m app.app
```

Aplikasi akan berjalan di: `http://localhost:5001`

---

## 📌 API Endpoints

| Endpoint                | Method | Deskripsi                                                                           |
| ----------------------- | ------ | ----------------------------------------------------------------------------------- |
| `/health`               | GET    | Cek status aplikasi                                                                 |
| `/input-docs`           | POST   | Upload dokumen (.md) *(params: `session_id`, `files`)*                              |
| `/ask`                  | POST   | Ajukan pertanyaan umum *(params: `session_id`, `question`)*                         |
| `/askproject`           | POST   | Ajukan pertanyaan terkait proyek *(params: `session_id`, `question`)* |
| `/conversation/history` | GET    | Ambil riwayat percakapan *(query param: `session_id`)*                              |
| `/conversation/clear`   | POST   | Hapus memory percakapan *(body: `session_id`)*                                      |
| `/documents/stats`      | GET    | Statistik dokumen                                                                   |
| `/documents/search`     | POST   | Pencarian dalam dokumen *(params: `session_id`, `query`)*                           |
| `/documents/reset`      | POST   | Hapus semua dokumen dari sistem                                                     |

---

## 💡 Contoh Penggunaan

### Upload Dokumen

```bash
curl -X POST http://localhost:5001/input-docs \
  -F "files=@sample_docs/panduan_aktuaria.md" \
  -F "files=@sample_docs/laporan_keuangan.md" \
  -F "files=@sample_docs/rumus_aktuaria.md" \
  -F "session_id=test_session"
```

### Ajukan Pertanyaan Umum

```bash
curl -X POST http://localhost:5001/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa itu prinsip aktuaria?",
    "session_id": "test_session"
  }'
```

### Ajukan Pertanyaan Proyek

```bash
curl -X POST http://localhost:5001/askproject \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa hasil analisis sensitivitas proyek A?",
    "session_id": "test_session"
  }'
```

---

## 🗂️ Struktur Project

```
actuarial-chatbot/
├── app/                    # Aplikasi utama
│   ├── main.py             # Flask entry point
│   ├── config.py           # Konfigurasi aplikasi
│   ├── models/             # Struktur data
│   ├── services/           # Logika bisnis
│   └── utils/              # Fungsi pembantu
├── sample_docs/            # Contoh dokumen markdown
├── data/                   # Penyimpanan data lokal
├── requirements.txt        # Daftar dependencies
└── README.md               # Dokumentasi
```

---

## 🔧 Kustomisasi

* **Menambah Tipe Dokumen Baru:**
  Edit `app/services/document_processor.py` pada method `_extract_document_type()`.

* **Ubah Prompt AI:**
  Modifikasi `app/services/chat_service.py` pada method `_get_custom_prompt_template()`.

* **Atur Chunk Size:**
  Edit `CHUNK_SIZE` dan `CHUNK_OVERLAP` pada `app/config.py`.

---

## 🛠️ Troubleshooting

**Masalah Umum:**

| Masalah          | Solusi                                                     |
| ---------------- | ---------------------------------------------------------- |
| OpenAI API Error | Pastikan API key valid dan masih memiliki kredit           |
| ChromaDB Error   | Hapus folder `data/vectorstore` dan restart aplikasi       |
| Memory Error     | Kurangi nilai `CHUNK_SIZE` di konfigurasi                  |
| Import Error     | Pastikan semua dependensi ter-install dengan `pip install` |

**Log Debug:**
Lihat file `app.log` di root folder untuk informasi detail saat error.

---

## 🚀 How to Deploy to Azure (Manual)

### ✅ 1. Persiapan

Pastikan sudah menginstall:

* [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
* [Docker](https://docs.docker.com/get-docker/)

Login ke akun Azure:

```bash
az login
```

---

### ✅ 2. Buat Resource Group

```bash
az group create --name chatbot-rg --location southeastasia
```

---

### ✅ 3. Buat Azure Container Registry (ACR)

> Gunakan nama yang unik!

```bash
az acr create --resource-group chatbot-rg --name chatbotregistryaktuaria --sku Basic
az acr login --name chatbotregistryaktuaria
```

---

### ✅ 4. Build & Push Docker Image

> **Penting:** gunakan arsitektur `linux/amd64` agar kompatibel dengan Azure App Service.

```bash
docker buildx create --use
docker buildx build --platform linux/amd64 -t chatbotregistryaktuaria.azurecr.io/chatbot-app:latest --push .
```

---

### ✅ 5. Buat App Service Plan

```bash
az appservice plan create \
  --name chatbot-app-plan \
  --resource-group chatbot-rg \
  --is-linux \
  --sku B1
```

---

### ✅ 6. Buat Web App berbasis Docker

```bash
az webapp create \
  --resource-group chatbot-rg \
  --plan chatbot-app-plan \
  --name chatbot-app-service \
  --deployment-container-image-name chatbotregistryaktuaria.azurecr.io/chatbot-app:latest
```

---

### ✅ 7. Set Docker Registry Credentials

Dapatkan username dan password ACR:

```bash
az acr credential show --name chatbotregistryaktuaria
```

Lalu set credentials ke Web App:

```bash
az webapp config container set \
  --name chatbot-app-service \
  --resource-group chatbot-rg \
  --docker-custom-image-name chatbotregistryaktuaria.azurecr.io/chatbot-app:latest \
  --docker-registry-server-url https://chatbotregistryaktuaria.azurecr.io \
  --docker-registry-server-user <username> \
  --docker-registry-server-password <password>
```

---

### ✅ 8. Set Environment Variables

```bash
az webapp config appsettings set \
  --name chatbot-app-service \
  --resource-group chatbot-rg \
  --settings \
  OPENAI_API_KEY=<your-api-key> \
  FLASK_ENV=development \
  FLASK_DEBUG=True \
  CHROMA_DB_PATH=./data/vectorstore \
  LOG_LEVEL=INFO
```

---

### ✅ 9. Restart Web App

```bash
az webapp restart \
  --name chatbot-app-service \
  --resource-group chatbot-rg
```

---

### ✅ 10. Akses Aplikasi

Kunjungi di browser:

```
https://chatbot-app-service.azurewebsites.net
```

---

### ❗ Tips Debugging

Lihat live log untuk debugging:

```bash
az webapp log tail \
  --name chatbot-app-service \
  --resource-group chatbot-rg
```

---

## 🔁 How to Update / Redeploy

Jika kamu melakukan update pada file atau kode:

### 1. Update File Lokal

* Edit file Python, konfigurasi, atau lainnya.
* Tambahkan dependensi baru jika perlu.

### 2. Rebuild dan Push Ulang Docker Image

```bash
docker buildx build --platform linux/amd64 -t chatbotregistryaktuaria.azurecr.io/chatbot-app:latest --push .
```

### 3. Restart Web App

```bash
az webapp restart \
  --name chatbot-app-service \
  --resource-group chatbot-rg
```

> Azure akan menarik (pull) image terbaru dari ACR dan menjalankan versi terbaru.

### (Opsional) Lihat Log Live Saat Redeploy

```bash
az webapp log tail \
  --name chatbot-app-service \
  --resource-group chatbot-rg
```

---

### 📦 Tips Tambahan

* Untuk deployment versi: gunakan tag seperti `chatbot-app:v1`, `v2`, dst.
* Ingin staging vs production? Gunakan **deployment slots** di Azure App Service.
* Untuk otomasi, gunakan **GitHub Actions CI/CD** — saya siap bantu kalau kamu ingin integrasi penuh otomatis.

## 📄 License

Internal Use Only - PT Datapolis Indonesia
