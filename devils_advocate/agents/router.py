"""Router node - decides whether to drill down or ask new question."""

from devils_advocate.models.state import InterviewState
from devils_advocate.config import SIMILARITY_THRESHOLD


def route_based_on_answer(state: InterviewState) -> str:
    """Route to drill_down or question_generator based on answer quality.

    Args:
        state: Current interview state

    Returns:
        "drill_down" if answer was weak (and not already in drill-down),
        "question_generator" otherwise
    """

    similarity_score = state.get("last_similarity_score", 0.5)
    in_drilldown = state.get("in_drilldown_mode", False)

    # If already in drill-down, move to next question regardless of score
    if in_drilldown:
        return "question_generator"

    # Weak answer - need to probe deeper (first time only)
    if similarity_score < SIMILARITY_THRESHOLD:
        return "drill_down"

    # Strong answer - move to next question
    return "question_generator"
