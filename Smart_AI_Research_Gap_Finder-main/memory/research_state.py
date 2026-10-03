from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class ResearchState:
    papers: List[Dict[str, Any]] = field(default_factory=list)
    summaries: str = ''
    comparison: str = ''
    review: str = ''
    trends: str = ''
    potential_gaps: str = ''
    verified_gaps: str = ''
    research_ideas: str = ''
    proposal: str = ''
    evidence: Dict[str, Any] = field(default_factory=dict)
    agent_log: List[str] = field(default_factory=list)
