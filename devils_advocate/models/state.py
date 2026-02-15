"""State models for LangGraph interview flow."""

from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage


class InterviewState(TypedDict):
    """State for the interview conversation graph."""

    # Context from JD
    role_title: str
    company_name: str
    company_values: List[str]
    required_skills: List[str]

    # Conversation history
    messages: Annotated[List[BaseMessage], operator.add]
    transcript: str  # Full text transcript for display

    # Logic control
    current_phase: str  # 'intro', 'behavioral', 'product_sense', 'feedback'
    skepticism_level: float  # 0.0 to 1.0
    question_count: int  # Track number of questions asked
    in_drilldown_mode: bool  # True if currently in drill-down, False otherwise

    # Scoring and evaluation
    question_evaluations: List[Dict[str, Any]]
    # Each dict: { "question": str, "user_answer": str, "flaw": str, "score": float, "suggestion": str }

    # Current answer analysis
    last_similarity_score: float  # Most recent answer similarity (0.0 to 1.0)
    last_user_answer: str  # Most recent user response
