from pathlib import Path
import hashlib
import uuid

from app.core.config import Settings
from app.services.embeddings_service import EmbeddingService
from app.services.llm_service import LocalLLM
from app.services.qdrant_service import QdrantService


# Faqat shu turdagi fayllar bazaga yuklanadi (mahsulot tavsiflari)
ALLOWED_SUFFIXES = {'.md', '.txt', '.json'}


# orchestrator
class RAGService:
    """CERANY mahsulotlari bazasidan qidirib, CeraGPT orqali javob yozadi."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        qdrant: QdrantService,
        llm: LocalLLM,
    ) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._qdrant = qdrant
        self._llm = llm

    # ---------- 1. Mahsulotlarni bazaga yuklash ----------
    def ingest(self, documents: list[dict]) -> int:
        return self._qdrant.upsert_documents(documents)

    def ingest_folder(self, path: str) -> tuple[int, int]:
        root = Path(path).expanduser()
        documents: list[dict] = []
        skipped = 0
        ignore_dirs = {'.git', '.venv', 'venv', '__pycache__', 'node_modules'}
        for file_path in root.rglob('*'):
            if not file_path.is_file():
                continue
            if any(part in ignore_dirs for part in file_path.parts):
                continue
            if file_path.suffix.lower() not in ALLOWED_SUFFIXES:
                skipped += 1
                continue
            try:
                text = file_path.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                skipped += 1
                continue
            if not text.strip():
                skipped += 1
                continue
            chunks = self._chunk_text(text)
            if not chunks:
                skipped += 1
                continue
            product = file_path.stem.replace('_', ' ').replace('-', ' ')
            for index, chunk in enumerate(chunks):
                documents.append(
                    {
                        'id': self._stable_id(file_path, chunk, index),
                        'text': chunk,
                        'metadata': {'path': str(file_path), 'product': product, 'chunk': index},
                    }
                )
        ingested = self._qdrant.upsert_documents(documents) if documents else 0
        return ingested, skipped

    # ---------- 2. Savolga javob berish ----------
    def answer(self, question: str) -> tuple[str, list[dict]]:
        filtered: list[dict] = []
        context = ''
        if self._settings.rag_enabled and self._settings.rag_mode != 'never':
            query_vector = self._embeddings.embed_texts([question])[0]
            sources = self._qdrant.search(query_vector, limit=self._settings.top_k)
            filtered = [s for s in sources if s['score'] >= self._settings.min_score]

            use_rag = self._settings.rag_mode == 'always'
            if self._settings.rag_mode == 'auto':
                use_rag = not self._is_small_talk(question)
                if filtered:
                    top_score = max(s['score'] for s in filtered)
                    if top_score < self._settings.min_score_strict:
                        use_rag = False

            if use_rag and filtered:
                max_sources = min(self._settings.top_k, self._settings.max_sources)
                filtered = filtered[:max_sources]
                context = self._build_context(filtered)
            else:
                filtered = []
                context = ''

        answer = self._llm.generate(self._build_prompt(question, context))
        return self._clean_answer(answer), filtered

    def _build_context(self, sources: list[dict]) -> str:
        chunks = []
        total = 0
        for index, source in enumerate(sources, start=1):
            text = source['text']
            if not text:
                continue
            product = (source.get('metadata') or {}).get('product')
            if product:
                text = f'[{index}] {product}\n{text}'
            if total + len(text) > self._settings.max_context_chars:
                break
            chunks.append(text)
            total += len(text)
        return '\n\n'.join(chunks)

    def _build_prompt(self, question: str, context: str) -> str:
        # CeraGPT "### Instruction / ### Response" formatida o'qitilgan,
        # shuning uchun prompt ham aynan shu formatda bo'lishi kerak.
        if context:
            input_block = f'CERANY product information:\n{context}'
            rules = (
                'You are CeraGPT, the skincare assistant of CERANY (a K-beauty store). '
                'Answer the customer question using the CERANY product information when relevant. '
                'Only recommend products that appear in the product information. '
                'Keep the answer to 2-5 sentences.'
            )
            return (
                f'### Instruction:\n{rules}\n\nQuestion: {question}\n\n'
                f'### Input:\n{input_block}\n\n'
                '### Response:\n'
            )
        rules = (
            'You are CeraGPT, the skincare assistant of CERANY (a K-beauty store). '
            'Answer briefly and helpfully in 2-5 sentences. '
            'Do not invent product names.'
        )
        return f'### Instruction:\n{rules}\n\nQuestion: {question}\n\n### Response:\n'

    def _chunk_text(self, text: str) -> list[str]:
        size = max(1, self._settings.chunk_size_chars)
        overlap = max(0, min(self._settings.chunk_overlap_chars, size - 1))
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            start = end - overlap
        return chunks

    def _stable_id(self, file_path: Path, text: str, index: int) -> str:
        digest = hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f'{file_path}:{index}:{digest}'))

    def _clean_answer(self, answer: str) -> str:
        for marker in ('### Instruction', '### Input', '### Response', '\nQuestion:', 'Question:'):
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        return answer.strip()

    def _is_small_talk(self, question: str) -> bool:
        """Salomlashish yoki "sen kimsan" kabi savollar uchun bazadan qidirish shart emas."""
        q = question.lower().strip()
        small_talk = (
            'hello', 'hi ', 'hey', 'who are you', 'what is your name',
            'thank', 'good morning', 'good evening', 'bye',
        )
        return any(q.startswith(m) or m in q for m in small_talk) and len(q) < 40
