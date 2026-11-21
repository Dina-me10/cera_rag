from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

from app.api.routes import router
from app.core.config import Settings
from app.services.embeddings_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.llm_service import LocalLLM
from app.services.rag_service import RAGService

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("*** CERA RAG STARTS ***")
    settings = Settings()
    embeddings = EmbeddingService(settings.embedding_model)
    qdrant = QdrantService(settings, embeddings)
    logger.info("Qdrant connected, collection '%s' is ready", settings.qdrant_collection)
    llm = LocalLLM(settings)
    rag = RAGService(settings, embeddings, qdrant, llm)

    app.state.settings = settings
    logging.basicConfig(
        level=settings.log_level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    app.state.embeddings = embeddings
    app.state.qdrant = qdrant
    app.state.llm = llm
    app.state.rag = rag
    yield


app = FastAPI(title='CERA RAG', version='1.0.0', lifespan=lifespan)


app.include_router(router)
