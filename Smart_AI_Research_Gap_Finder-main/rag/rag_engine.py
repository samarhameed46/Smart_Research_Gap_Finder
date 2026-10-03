import re
from pathlib import Path
from typing import Dict, List

import fitz
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, TOP_K


class AdvancedRAGEngine:
    """Structure-aware PDF RAG with metadata, MMR retrieval and lightweight lexical reranking."""

    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.documents: List[Dict] = []

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_documents(self, pdf_paths: List[str]) -> List[Dict]:
        records: List[Dict] = []
        for path in pdf_paths:
            doc = fitz.open(path)
            title = Path(path).stem
            for page_no, page in enumerate(doc, start=1):
                text = self._clean(page.get_text())
                if text:
                    records.append({'text': text, 'source': title, 'page': page_no})
            doc.close()
        if not records:
            raise ValueError('No readable text was found in the uploaded PDFs.')
        return records

    def create_chunks(self, records: List[Dict]) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=180,
            separators=['\n\n', '. ', '; ', ', ', ' ']
        )
        chunks: List[Dict] = []
        for record in records:
            for idx, chunk in enumerate(splitter.split_text(record['text'])):
                chunks.append({**record, 'text': chunk, 'chunk_id': idx})
        return chunks

    def build(self, pdf_paths: List[str]) -> None:
        records = self.extract_documents(pdf_paths)
        self.documents = self.create_chunks(records)
        texts = [d['text'] for d in self.documents]
        metadatas = [{k: d[k] for k in ('source', 'page', 'chunk_id')} for d in self.documents]
        self.vector_store = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        if self.vector_store is None:
            raise ValueError('Please process papers before retrieval.')
        docs = self.vector_store.max_marginal_relevance_search(query, k=min(k, len(self.documents)), fetch_k=min(max(k * 3, 12), len(self.documents)))
        terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        scored = []
        for doc in docs:
            text_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', doc.page_content.lower()))
            lexical = len(terms & text_terms)
            scored.append((lexical, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {'text': d.page_content, 'source': d.metadata.get('source', 'Unknown'), 'page': d.metadata.get('page', '?'), 'chunk_id': d.metadata.get('chunk_id', '?')}
            for _, d in scored
        ]

    def retrieve_context(self, query: str, k: int = TOP_K) -> str:
        results = self.retrieve(query, k)
        return '\n\n'.join(f"[Source: {r['source']}, page {r['page']}]\n{r['text']}" for r in results)
