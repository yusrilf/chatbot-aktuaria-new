# LangGraph Integration Setup Guide

## Overview
LangGraph RAG workflow telah berhasil diintegrasikan ke backend Flask. Workflow ini menyediakan sistem RAG yang lebih canggih dengan fitur:
- Query generation dan response adaptif
- Document retrieval menggunakan Pinecone
- Document relevance grading
- Query rewriting untuk hasil yang lebih baik
- Answer generation

## Files yang Diubah/Dibuat

### 1. Service Module
**File:** `app/services/langgraph_service.py`
- Service class untuk LangGraph RAG workflow
- Menggunakan GPT-4o-mini untuk response generation
- Menggunakan GPT-4o untuk document grading
- Integrasi dengan Pinecone vector store
- Health check functionality

### 2. Routes
**File:** `app/routes/langgraph_routes.py`
- Blueprint untuk LangGraph endpoints
- 3 endpoints:
  - `POST /langgraph/ask` - Process question through RAG workflow
  - `GET /langgraph/health` - Check service health
  - `GET /langgraph/info` - Get service information

### 3. App Factory
**File:** `app/app_factory.py`
- Tambahan import `langgraph_bp`
- Register blueprint di `register_blueprints()`
- Update root endpoint untuk list new endpoints

### 4. Requirements
**File:** `requirements.txt`
- Tambahan dependency: `langgraph>=0.2.0`

## Environment Variables

Pastikan file `.env` memiliki variable berikut:

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration (required for LangGraph)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-actuaria-3  # atau nama index yang sesuai

# Optional Pinecone settings
PINECONE_NAMESPACE=default
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

**PENTING:** Pastikan Pinecone index sudah dibuat dengan:
- Dimension: 1536 (untuk text-embedding-3-small)
- Metric: cosine
- Cloud: AWS (us-east-1)

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Environment Variables

```bash
# Check if required env vars are set
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

required = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
for var in required:
    value = os.getenv(var)
    if value:
        print(f'✓ {var} is set')
    else:
        print(f'✗ {var} is NOT set')
"
```

### 3. Run Flask Server

```bash
# Development
python app/main.py

# Or using Flask command
flask --app app.main run --host=0.0.0.0 --port=5001
```

## Testing Endpoints

### 1. Check Service Health

```bash
curl http://localhost:5001/langgraph/health
```

Expected response:
```json
{
  "success": true,
  "message": "LangGraph service is healthy",
  "data": {
    "status": "healthy",
    "components": {
      "models": "ok",
      "retriever": "ok",
      "graph": "ok",
      "pinecone": "ok"
    },
    "index_name": "rag-actuaria-3",
    "document_count": 1234
  }
}
```

### 2. Get Service Info

```bash
curl http://localhost:5001/langgraph/info
```

### 3. Ask Question (Main Endpoint)

```bash
curl -X POST http://localhost:5001/langgraph/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Apa itu PSAK 219?"
  }'
```

Expected response:
```json
{
  "success": true,
  "message": "Query processed successfully",
  "data": {
    "answer": "PSAK 219 adalah...",
    "question": "Apa itu PSAK 219?",
    "metadata": {
      "service": "langgraph",
      "workflow": "rag_with_grading"
    }
  }
}
```

## Docker Deployment

### 1. Build Docker Image

```bash
docker build -t chatbot-aktuaria:latest .
```

### 2. Run Container

```bash
docker run -d \
  --name chatbot-aktuaria \
  -p 5000:5000 \
  -p 8501:8501 \
  -e OPENAI_API_KEY="your_key" \
  -e PINECONE_API_KEY="your_key" \
  -e PINECONE_INDEX_NAME="rag-actuaria-3" \
  chatbot-aktuaria:latest
```

### 3. Verify in Docker

```bash
# Check logs
docker logs chatbot-aktuaria

# Test endpoint
curl http://localhost:5000/langgraph/health
```

## Azure Deployment

### Prerequisites
1. Azure Container Registry (ACR) atau Docker Hub
2. Azure App Service atau Azure Container Instances
3. Environment variables configured in Azure

### Steps

#### 1. Push Image to Registry

```bash
# Tag image
docker tag chatbot-aktuaria:latest <registry>/chatbot-aktuaria:latest

# Push to registry
docker push <registry>/chatbot-aktuaria:latest
```

#### 2. Deploy to Azure App Service

Via Azure Portal:
1. Create App Service (Container)
2. Set Docker image: `<registry>/chatbot-aktuaria:latest`
3. Configure Application Settings:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
   - `PORT=5000`
4. Set health check path: `/health`
5. Deploy

Via Azure CLI:

```bash
az webapp create \
  --resource-group <resource-group> \
  --plan <app-service-plan> \
  --name <app-name> \
  --deployment-container-image-name <registry>/chatbot-aktuaria:latest

# Set environment variables
az webapp config appsettings set \
  --resource-group <resource-group> \
  --name <app-name> \
  --settings \
    OPENAI_API_KEY="your_key" \
    PINECONE_API_KEY="your_key" \
    PINECONE_INDEX_NAME="rag-actuaria-3" \
    PORT=5000
```

#### 3. Verify Deployment

```bash
# Check health endpoint
curl https://<app-name>.azurewebsites.net/health

# Check LangGraph health
curl https://<app-name>.azurewebsites.net/langgraph/health
```

## Troubleshooting

### Issue: "LangGraph service initialization failed"

**Cause:** Missing or invalid PINECONE_API_KEY

**Solution:**
```bash
# Verify Pinecone API key
python -c "
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print('✓ Pinecone connection OK')
print('Indexes:', pc.list_indexes())
"
```

### Issue: "Index not found"

**Cause:** PINECONE_INDEX_NAME tidak sesuai atau index belum dibuat

**Solution:**
```bash
# List available indexes
python -c "
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
indexes = pc.list_indexes()
print('Available indexes:')
for idx in indexes:
    print(f'  - {idx.name}')
"
```

### Issue: "ModuleNotFoundError: No module named 'langgraph'"

**Cause:** Dependencies belum terinstall

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Import Error untuk create_retriever_tool

**Cause:** Versi langchain tidak kompatibel

**Solution:**
Sudah diatasi dengan menggunakan import path yang benar:
```python
from langchain.tools.retriever import create_retriever_tool
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│              LangGraph RAG Workflow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. Generate Query or Respond                   │
│      ↓                                          │
│  2. Retrieve (if needed) ────→ Pinecone        │
│      ↓                                          │
│  3. Grade Documents                             │
│      ├─ Relevant? → Generate Answer             │
│      └─ Not Relevant? → Rewrite Question → (1) │
│                                                 │
└─────────────────────────────────────────────────┘
```

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/langgraph/ask` | POST | Process question through RAG workflow |
| `/langgraph/health` | GET | Check service health |
| `/langgraph/info` | GET | Get service information |
| `/health` | GET | Overall application health |
| `/` | GET | API root with all endpoints |

## Next Steps

1. ✅ Setup complete
2. ⏳ Test endpoints locally
3. ⏳ Deploy to Azure
4. ⏳ Monitor performance
5. ⏳ Add additional features (e.g., streaming responses)

## Support

Jika ada issue:
1. Check logs: `docker logs chatbot-aktuaria` atau Azure App Service Logs
2. Verify environment variables
3. Test Pinecone connection
4. Check `/langgraph/health` endpoint
