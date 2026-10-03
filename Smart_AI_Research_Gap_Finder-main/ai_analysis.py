
# ai_analysis.py

import logging
import os
from typing import Dict

from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAnalysis:
    """
    Handles:
    - Summarization
    - Trend Analysis
    - Limitation Extraction
    - Research Gap Detection
    - Proposal Generation
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not found.")

        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-20b"

    def _generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content

            if not content:
                return "No information generated."

            return content

        except Exception as e:
            logger.error(f"Groq API Error: {e}")
            raise

    def summarize_papers(self, context: str) -> str:
        prompt = f"""
        Summarize the following research papers.

        Context:
        {context}
        """
        return self._generate(prompt)

    def extract_limitations(self, context: str) -> str:
        prompt = f"""
        You are an expert research analyst.

        Analyze the research papers and extract:

        1. Limitations
        2. Challenges
        3. Weaknesses
        4. Future work suggestions

        Return at least 10 detailed bullet points.

        Context:

        {context}
        """
        return self._generate(prompt)

    def detect_trends(self, context: str) -> str:
        prompt = f"""
        Analyze the research trends, common methods,
        datasets, and technologies used.

        {context}
        """
        return self._generate(prompt)

    def find_research_gaps(self, context: str) -> str:
        prompt = f"""
        You are an expert research gap finder.

        Analyze all papers and identify:

        1. Missing research areas
        2. Unsolved problems
        3. Weakly explored topics
        4. Future opportunities
        5. Novel directions

        Return at least 10 detailed research gaps.

        Context:

        {context}
        """
        return self._generate(prompt)

    def generate_proposal(self, research_gaps: str) -> str:
        prompt = f"""
        Generate a research proposal idea using the
        discovered research gaps below.

        Research Gaps:
        {research_gaps}
        """
        return self._generate(prompt)

    def run_full_analysis(self, context: str) -> Dict[str, str]:
        summary = self.summarize_papers(context)
        limitations = self.extract_limitations(context)
        trends = self.detect_trends(context)
        research_gaps = self.find_research_gaps(context)
        proposal = self.generate_proposal(research_gaps)

        print("SUMMARY:", summary[:200])
        print("LIMITATIONS:", limitations[:200])
        print("TRENDS:", trends[:200])
        print("RESEARCH GAPS:", research_gaps[:200])
        print("PROPOSAL:", proposal[:200])

        return {
            "summary": summary,
            "limitations": limitations,
            "trends": trends,
            "research_gaps": research_gaps,
            "proposal": proposal,
        }

