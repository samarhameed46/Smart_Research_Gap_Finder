import os
from typing import Optional
from groq import Groq
from config import MODEL_NAME, MAX_CONTEXT_CHARS, get_groq_api_key

class BaseAgent:
    name = 'Agent'
    def __init__(self, rag=None):
        self.rag = rag
        self.client = Groq(api_key=get_groq_api_key())
        self.model = MODEL_NAME

    def retrieve(self, query: str, k: int = 8) -> str:
        return self.rag.retrieve_context(query, k) if self.rag else ''

    def ask(self, instruction: str, context: str = '', temperature: float = 0.2) -> str:
        context = context[:MAX_CONTEXT_CHARS]
        prompt = f"""You are {self.name}, part of a research-analysis multi-agent system.\n\nRules:\n- Be evidence-based and distinguish evidence from inference.\n- Do not claim a research gap is definitely novel. Call it a potential research opportunity unless verified.\n- Cite uploaded evidence using [Source: filename, page N] when available.\n\nTASK:\n{instruction}\n\nCONTEXT:\n{context}"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'system', 'content': 'You are a careful academic research assistant.'}, {'role': 'user', 'content': prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ''
