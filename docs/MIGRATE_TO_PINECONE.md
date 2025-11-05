# Migrasi ke Pinecone: Panduan End-to-End

Dokumen ini memandu migrasi penyimpanan vektor dari Chroma (lokal) ke Pinecone (serverless). Migrasi ini dirancang untuk meningkatkan stabilitas di beban tinggi, menghapus risiko "database is locked", dan menyederhanakan operasi produksi.

## Ringkasan
- Backend vektor akan ditentukan melalui `VECTOR_BACKEND` di `.env`.
- Tetap gunakan OpenAI embeddings (`text-embedding-3-large` disarankan), sesuaikan dimensi index Pinecone.
- Gunakan integrasi LangChain `PineconeVectorStore` untuk minim perubahan di kode.
- Rebuild index dari sumber dokumen, bukan mengekspor dari Chroma.

## Prasyarat
- Akun Pinecone aktif dan API Key.
- Akses untuk membuat index serverless (cloud dan region).
- OpenAI API Key valid.

## Dependensi
Tambahkan paket berikut (opsi A: via pip langsung, opsi B: lewat `requirements.txt`).

- `pip install pinecone-client>=3.0.0 langchain-pinecone>=0.0.3`
- Atau tambahkan ke `requirements.txt`:
  - `pinecone-client>=3.0.0`
  - `langchain-pinecone>=0.0.3`

## Konfigurasi .env
Gunakan `.env.example` terbaru sebagai referensi, lalu buat `.env` dengan nilai produksi.

Variabel utama:
- `VECTOR_BACKEND=pinecone` untuk mengaktifkan Pinecone.
- `PINECONE_API_KEY` set API key Pinecone.
- `PINECONE_INDEX_NAME` nama index (misal: `aktuaria-docs`).
- `PINECONE_NAMESPACE` namespace (misal: `default`).
- `PINECONE_CLOUD` cloud saat membuat index (misal: `aws`).
- `PINECONE_REGION` region serverless (misal: `us-east-1`).
- `PINECONE_DIMENSION` sesuaikan dengan model embedding:
  - `text-embedding-3-large` -> `3072`
  - `text-embedding-3-small` -> `1536`
- `PINECONE_METRIC=cosine` (disarankan untuk OpenAI embeddings).

Contoh `.env` minimal:
```
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-large
VECTOR_BACKEND=pinecone
PINECONE_API_KEY=pc-...
PINECONE_INDEX_NAME=aktuaria-docs
PINECONE_NAMESPACE=default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_DIMENSION=3072
PINECONE_METRIC=cosine
```

## Perubahan Konfigurasi Kode
Ubah `app/config.py` agar membaca variabel Pinecone dan toggle backend dari `.env`.

Di dalam class `Config`, tambahkan/ubah:
```python
# app/config.py
class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')

    # Backend toggle
    VECTOR_BACKEND = os.getenv('VECTOR_BACKEND', 'chroma')

    # Chroma
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './data/vectorstore')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'actuarial_documents')

    # Pinecone
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'aktuaria-docs')
    PINECONE_NAMESPACE = os.getenv('PINECONE_NAMESPACE', 'default')
    PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
    PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
    PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '3072'))
    PINECONE_METRIC = os.getenv('PINECONE_METRIC', 'cosine')
```

## Integrasi di VectorStoreManager
Tambahkan dukungan Pinecone di `app/models/embeddings/vector_store_manager.py` pada metode `_initialize_vectorstore`.

Contoh implementasi (ringkas):
```python
# app/models/embeddings/vector_store_manager.py
from app.config import config
from langchain_openai import OpenAIEmbeddings

def _initialize_vectorstore(self) -> None:
    if config.VECTOR_BACKEND == 'pinecone':
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore

        # Init Pinecone client
        pc = Pinecone(api_key=config.PINECONE_API_KEY)

        # Ensure index exists (opsional, bisa dibuat via console)
        existing = {i.name for i in pc.list_indexes()}
        if config.PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.PINECONE_DIMENSION,
                metric=config.PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION
                )
            )

        # LangChain vectorstore wrapper
        self.vectorstore = PineconeVectorStore(
            index_name=config.PINECONE_INDEX_NAME,
            namespace=config.PINECONE_NAMESPACE,
            embedding=self.embeddings,
        )
    else:
        # Chroma (eksisting)
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
        os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True, is_persistent=True)
        )
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings
        )
```

Catatan:
- Semua pemanggilan `add_documents`, `similarity_search`, dan `similarity_search_with_score` tetap kompatibel dengan `PineconeVectorStore`.
- Hindari membuat klien/instans vektorstore duplikat di layanan lain (mis. `EnhancedEmbeddingService`): gunakan satu `VectorStoreManager`.

## Refaktor Layanan Embedding
`app/services/enhanced_embedding_service.py` saat ini membuat klien Chroma sendiri. Untuk Pinecone:
- Ambil `VectorStoreManager` dari singleton/service manager.
- Gunakan `vector_store_manager.vectorstore.add_documents(...)` untuk menyimpan chunk.

Sketsa perubahan:
```python
# app/services/enhanced_embedding_service.py
from app.models.embeddings import VectorStoreManager
from app.utils.singleton_manager import get_service_manager

class EnhancedEmbeddingService:
    def __init__(...):
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model, openai_api_key=config.OPENAI_API_KEY)
        # Ambil vectorstore terpusat
        vsm = get_service_manager().get_service('vector_store_manager') or VectorStoreManager()
        self.vectorstore = vsm.vectorstore
        # ... lanjutkan seperti biasa (preprocessor, chunking, dsb.)
```

Dengan ini, tidak ada klien Chroma/Pinecone yang dibuat duplikat di banyak tempat.

## Migrasi Data (Rebuild Index)
Alih-alih mengekspor dari Chroma, rebuild dari sumber dokumen agar bersih dan konsisten.

Langkah:
1. Pastikan `.env` sudah di-set ke `VECTOR_BACKEND=pinecone` dan variabel Pinecone benar.
2. Pastikan dependensi Pinecone terpasang.
3. Jalankan proses embedding ulang dari direktori sumber:
   - `sample_docs/` (global docs)
   - `data/uploaded_documents/` (dokumen unggahan)

Contoh skrip sederhana (opsional):
```python
# scripts/migrate_to_pinecone.py (opsional)
from pathlib import Path
from app.models.embeddings import VectorStoreManager
from app.services.global_docs_preprocessor import GlobalDocsPreprocessor

vsm = VectorStoreManager()
pre = GlobalDocsPreprocessor()

paths = list(Path('sample_docs').rglob('*.md')) + list(Path('data/uploaded_documents').rglob('*.md'))
docs = []
for p in paths:
    docs.extend(pre.process_file(str(p)))  # hasil berupa List[Document]

vsm.vectorstore.add_documents(docs)
print(f"Upserted {len(docs)} chunks to Pinecone")
```

## Verifikasi
- Jalankan aplikasi: `python -m app.main` atau via `make run` jika tersedia.
- Coba endpoint/fitur pencarian dan tanya jawab.
- Periksa Pinecone Console untuk memastikan index menerima data (jumlah vector bertambah).

## Rollback
- Set `VECTOR_BACKEND=chroma` di `.env` untuk kembali ke Chroma.
- Tidak perlu menghapus index Pinecone; biarkan sebagai cadangan.

## Performa & Biaya
- Pinecone menambah latensi jaringan dibanding lokal, tetapi stabil di beban tinggi.
- Batasi paralelisme write (`max_workers`) bila perlu untuk mengontrol biaya dan beban.

## Known Differences
- Filter metadata tetap didukung, namun semantik filter Pinecone dapat berbeda dari Chroma.
- `persist()` tidak relevan di Pinecone; data tersimpan otomatis.

## Checklist Migrasi
- [ ] Menambahkan dependensi Pinecone.
- [ ] Update `.env` dengan variabel Pinecone dan `VECTOR_BACKEND=pinecone`.
- [ ] Update `app/config.py` membaca variabel baru.
- [ ] Update `VectorStoreManager` mendukung Pinecone.
- [ ] Refaktor layanan yang membuat klien Chroma langsung (gunakan `VectorStoreManager`).
- [ ] Rebuild index dari sumber dokumen.
- [ ] Verifikasi di aplikasi dan Pinecone Console.

## Keamanan
- Jangan commit API keys (`.env` sudah di-ignore).
- Rotasi key secara berkala bila diperlukan.

## Appendix: Dimensi Embedding
- OpenAI `text-embedding-3-large` = 3072
- OpenAI `text-embedding-3-small` = 1536