from agents.base import BaseAgent

class ResearchChatbot(BaseAgent):
    name = 'Research Assistant Agent'
    def ask(self, question: str) -> str:
        context = self.retrieve(question, 8)
        return self.ask_with_context(question, context)
    def ask_with_context(self, question, context):
        return super().ask(f'Answer the user question: {question}. Use only the supplied paper evidence. If evidence is insufficient, say so clearly.', context, 0.1)
