"""Pydantic schemas for data validation."""

from pydantic import BaseModel, Field
from typing import List


class JobContext(BaseModel):
    """Extracted context from a job description."""

    role_title: str = Field(description="Job title or role name")
    company_name: str = Field(description="Company name")
    company_values: List[str] = Field(
        description="Core company values mentioned in JD",
        default_factory=list
    )
    required_skills: List[str] = Field(
        description="Required technical or soft skills",
        default_factory=list
    )


class QuestionEvaluation(BaseModel):
    """Evaluation of a single interview question-answer pair."""

    question: str = Field(description="The question asked")
    user_answer: str = Field(description="User's response")
    flaw: str = Field(description="Key weakness identified in the answer")
    score: float = Field(description="Similarity score (0.0 to 1.0)", ge=0.0, le=1.0)
    suggestion: str = Field(description="How to improve the answer")


class IdealIntent(BaseModel):
    """Ideal answer pattern for semantic comparison."""

    category: str = Field(description="Category: 'behavioral', 'product_sense', 'technical'")
    intent: str = Field(description="The ideal intent or pattern to match")
    examples: List[str] = Field(
        description="Example phrases that match this intent",
        default_factory=list
    )
