# CERA RAG API

Simple, production-minded RAG service for CERANY skincare products, using FastAPI, Pydantic, Qdrant, and TinyLlama LLM (CeraGPT).

## Setup
1. Create a virtual environment and install dependencies:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Start the server:

```bash
uvicorn app.main:app --reload
```