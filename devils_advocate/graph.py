"""LangGraph assembly - wires all nodes into interview state machine."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from devils_advocate.models.state import InterviewState
from devils_advocate.agents.analyzer import analyze_answer
from devils_advocate.agents.router import route_based_on_answer
from devils_advocate.agents.drill_down import drill_down
from devils_advocate.agents.question_generator import generate_question


def create_interview_graph(vector_store):
    """Create the interview conversation graph.

    Simplified flow:
    1. Generate question -> END
    2. User provides answer (external)
    3. Analyze answer -> route -> drill_down/question_generator -> END

    Args:
        vector_store: ChromaDB vector store with ideal intents

    Returns:
        Compiled LangGraph
    """

    # Create state graph
    graph = StateGraph(InterviewState)

    # Add nodes
    graph.add_node("question_generator", generate_question)
    graph.add_node("analyzer", lambda state: analyze_answer(state, vector_store))
    graph.add_node("drill_down", drill_down)

    # Define conditional routing function
    def route_after_analysis(
        state: InterviewState,
    ) -> Literal["drill_down", "question_generator", "__end__"]:
        """Route based on answer quality."""
        return route_based_on_answer(state)

    # Add edges
    # Entry point
    graph.set_entry_point("analyzer")

    # After analysis, route based on answer quality
    graph.add_conditional_edges(
        "analyzer",
        route_after_analysis,
        {
            "drill_down": "drill_down",
            "question_generator": "question_generator"
        }
    )

    # Both drill_down and question_generator end
    graph.add_edge("drill_down", END)
    graph.add_edge("question_generator", END)

    # Compile with memory
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    return compiled_graph


def initialize_interview_state(job_context) -> InterviewState:
    """Initialize interview state from extracted job context.

    Args:
        job_context: JobContext Pydantic model

    Returns:
        Initial InterviewState
    """

    return {
        "role_title": job_context.role_title,
        "company_name": job_context.company_name,
        "company_values": job_context.company_values,
        "required_skills": job_context.required_skills,
        "messages": [],
        "transcript": "",
        "current_phase": "intro",
        "skepticism_level": 0.5,
        "question_count": 0,
        "in_drilldown_mode": False,
        "question_evaluations": [],
        "last_similarity_score": 0.0,
        "last_user_answer": ""
    }
