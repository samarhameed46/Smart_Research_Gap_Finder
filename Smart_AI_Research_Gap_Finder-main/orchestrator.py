import logging
from typing import Dict, List
from rag import AdvancedRAGEngine
from memory.research_state import ResearchState
from agents import (PaperAnalysisAgent, ComparisonAgent, ReviewAgent, TrendAgent, GapAgent, VerificationAgent, IdeaAgent, ProposalAgent, ResearchManagerAgent)
from chatbot import ResearchChatbot

logger = logging.getLogger(__name__)

class ResearchGapFinderOrchestrator:
    """Coordinates an agentic multi-agent research workflow over an Advanced RAG index."""
    def __init__(self):
        self.rag_engine = AdvancedRAGEngine()
        self.state = ResearchState()
        self.manager = ResearchManagerAgent(self.rag_engine)
        self.agents = {
            'paper_analysis': PaperAnalysisAgent(self.rag_engine), 'comparison': ComparisonAgent(self.rag_engine),
            'critical_review': ReviewAgent(self.rag_engine), 'trends': TrendAgent(self.rag_engine),
            'gap_detection': GapAgent(self.rag_engine), 'verification': VerificationAgent(self.rag_engine),
            'research_ideas': IdeaAgent(self.rag_engine), 'proposal': ProposalAgent(self.rag_engine)
        }
        self.chatbot = ResearchChatbot(self.rag_engine)

    def process_papers(self, pdf_paths: List[str]) -> Dict[str, str]:
        if not pdf_paths: return {'error': 'No PDF files were provided.'}
        try:
            self.state = ResearchState()
            self.rag_engine.build(pdf_paths)
            self.state.papers = [{'source': x['source'], 'page': x['page']} for x in self.rag_engine.documents]
            context = self.rag_engine.retrieve_context('research objectives methods datasets results limitations future work trends', 10)
            self.state.agent_log.append('Advanced RAG index built')
            self.state.summaries = self.agents['paper_analysis'].run(context); self.state.agent_log.append('Paper Analysis Agent completed')
            self.state.comparison = self.agents['comparison'].run(context); self.state.agent_log.append('Comparison Agent completed')
            self.state.review = self.agents['critical_review'].run(context); self.state.agent_log.append('Critical Review Agent completed')
            self.state.trends = self.agents['trends'].run(context); self.state.agent_log.append('Trend Agent completed')
            self.state.potential_gaps = self.agents['gap_detection'].run(context); self.state.agent_log.append('Gap Agent completed')
            self.state.verified_gaps = self.agents['verification'].run(self.state.potential_gaps, context); self.state.agent_log.append('Verification Agent completed')
            self.state.research_ideas = self.agents['research_ideas'].run(self.state.verified_gaps, context); self.state.agent_log.append('Research Idea Agent completed')
            self.state.proposal = self.agents['proposal'].run(self.state.research_ideas, context); self.state.agent_log.append('Proposal Agent completed')
            return {'summary': self.state.summaries, 'comparison': self.state.comparison, 'review': self.state.review, 'trends': self.state.trends, 'limitations': self.state.review, 'research_gaps': self.state.potential_gaps, 'verified_gaps': self.state.verified_gaps, 'research_ideas': self.state.research_ideas, 'proposal': self.state.proposal}
        except Exception as e:
            logger.exception('Workflow failed')
            return {'error': str(e)}

    def ask_chatbot(self, question: str) -> str:
        return self.chatbot.ask(question)
