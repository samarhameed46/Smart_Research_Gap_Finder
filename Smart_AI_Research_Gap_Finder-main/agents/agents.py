from .base import BaseAgent

class PaperAnalysisAgent(BaseAgent):
    name = 'Paper Analysis Agent'
    def run(self, context):
        return self.ask('Extract objectives, methods, datasets, key findings, limitations and future work for the papers. Organize by paper where possible.', context)

class ComparisonAgent(BaseAgent):
    name = 'Literature Comparison Agent'
    def run(self, context):
        return self.ask('Compare the papers: methods, datasets, assumptions, findings, limitations and disagreements. Identify recurring patterns and meaningful differences.', context)

class ReviewAgent(BaseAgent):
    name = 'Critical Review Agent'
    def run(self, context):
        return self.ask('Critically review the literature. Identify methodological weaknesses, missing evaluations, inconsistent findings, underexplored populations/datasets and assumptions that deserve further testing.', context)

class TrendAgent(BaseAgent):
    name = 'Trend Analysis Agent'
    def run(self, context):
        return self.ask('Identify research themes and trends across the supplied literature. Separate strong recurring patterns from observations based on only one paper.', context)

class GapAgent(BaseAgent):
    name = 'Research Gap Agent'
    def run(self, context):
        return self.ask('Identify 3-5 potential research gaps/opportunities. For each give: evidence, observed limitation, what appears underexplored, why it matters, and a possible research direction. Do not claim novelty as fact.', context)

class VerificationAgent(BaseAgent):
    name = 'Verification Agent'
    def run(self, gaps, context):
        return self.ask('Audit the proposed gaps against the evidence. For each, mark Supported / Weakly Supported / Contradicted, explain the evidence, identify missing evidence, and state what a researcher should verify in recent literature before claiming novelty.', gaps + '\n\nLITERATURE EVIDENCE:\n' + context)

class IdeaAgent(BaseAgent):
    name = 'Research Idea Agent'
    def run(self, verified, context):
        return self.ask('Turn the verified opportunities into 3 concrete research directions. For each provide a title, research question, objectives, methodology, possible datasets, evaluation metrics and expected contribution.', verified + '\n\nEVIDENCE:\n' + context)

class ProposalAgent(BaseAgent):
    name = 'Proposal Agent'
    def run(self, ideas, context):
        return self.ask('Create a concise proposal draft based on the strongest evidence-supported direction. Include title, problem statement, objectives, research questions, methodology, expected results and contribution. Clearly label assumptions.', ideas + '\n\nSUPPORTING EVIDENCE:\n' + context)

class ResearchManagerAgent:
    name = 'Research Manager Agent'
    def plan(self):
        return ['paper_analysis', 'comparison', 'critical_review', 'trends', 'gap_detection', 'verification', 'research_ideas', 'proposal']
