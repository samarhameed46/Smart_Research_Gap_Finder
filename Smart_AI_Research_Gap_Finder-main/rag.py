import logging
from typing import List

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        self.vector_store = None

    def extract_text_from_pdfs(
        self,
        pdf_paths: List[str]
    ) -> str:

        combined_text = ""

        for pdf_path in pdf_paths:
            try:
                doc = fitz.open(pdf_path)

                for page in doc:
                    combined_text += page.get_text()

                doc.close()

            except Exception as e:
                logger.error(
                    f"Error reading {pdf_path}: {e}"
                )

        if not combined_text.strip():
            raise ValueError(
                "No text found in uploaded PDFs."
            )

        return combined_text

    def clean_text(
        self,
        text: str
    ) -> str:

        return " ".join(text.split())

    def create_chunks(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return splitter.split_text(text)

    def build_vector_store(
        self,
        chunks: List[str]
    ):

        self.vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )

        return self.vector_store

    def retrieve_context(
        self,
        query: str,
        k: int = 5
    ) -> str:

        if self.vector_store is None:
            raise ValueError(
                "Vector store has not been built."
            )

        docs = self.vector_store.similarity_search(
            query,
            k=k
        )

        return "\n\n".join(
            doc.page_content for doc in docs
        )

    def get_full_context(self, max_chars: int = 12000) -> str:
        """
        Return as much of the processed document text as reasonably
        fits in a single LLM prompt, instead of only the handful of
        chunks a single similarity search would return. Used for
        whole-document analysis tasks (summary, trends, limitations,
        gaps, proposal) where a single top-k retrieval would discard
        most of the uploaded papers.
        """

        if self.vector_store is None:
            raise ValueError(
                "Vector store has not been built."
            )

        all_chunks = [
            doc.page_content
            for doc in self.vector_store.docstore._dict.values()
        ]

        combined = "\n\n".join(all_chunks)

        if len(combined) > max_chars:
            combined = combined[:max_chars]

        return combined

    def process_pdfs(
        self,
        pdf_paths: List[str]
    ) -> None:

        text = self.extract_text_from_pdfs(
            pdf_paths
        )

        text = self.clean_text(text)

        chunks = self.create_chunks(text)

        self.build_vector_store(chunks)

        logger.info(
            "PDF processing completed successfully."
        )
