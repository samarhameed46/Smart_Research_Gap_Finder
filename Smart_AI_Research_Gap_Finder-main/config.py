import os

MODEL_NAME = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
TOP_K = int(os.getenv('RAG_TOP_K', '8'))
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '18000'))


def get_groq_api_key() -> str:
    key = os.getenv('GROQ_API_KEY')
    if not key:
        raise ValueError('GROQ_API_KEY is not configured. Add it to Streamlit Secrets or your environment.')
    return key
