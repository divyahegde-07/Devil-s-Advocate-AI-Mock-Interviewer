"""QuestionGenerator node - creates new interview questions based on phase and JD context."""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from devils_advocate.models.state import InterviewState
from devils_advocate.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_QUESTIONS


def create_question_generator_chain(phase: str):
    """Create chain for generating contextual interview questions."""

    # Phase-specific instructions
    phase_instructions = {
        "intro": """This is the INTRO phase. Ask a simple, warm-up question to help the candidate relax.
DO NOT ask technical questions. DO NOT ask about specific projects or technologies.
ONLY ask one of these types:
- "Tell me about yourself and your background."
- "What interests you about this role?"
- "Walk me through your career journey."
Keep it conversational, friendly, and non-technical.""",
        "behavioral": """This is the BEHAVIORAL phase. Ask about past experiences using the STAR format.
Focus on: leadership, conflict resolution, failure/learning, data-driven decisions, teamwork.
Example: "Tell me about a time when you had to make a difficult decision with limited information.""",
        "product_sense": """This is the PRODUCT SENSE phase. Test their product thinking and user empathy.
Focus on: user needs, prioritization, tradeoffs, metrics, problem-solving frameworks.
Example: "How would you improve [product feature]?" or "Design a solution for [user problem].""",
        "feedback": """This is the FEEDBACK phase. Wrap up and ask if they have questions.
Example: "Do you have any questions for me about the role or team?"""
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an experienced interviewer.

Company: {company_name}
Role: {role_title}
Company Values: {company_values}
Required Skills: {required_skills}

{phase_instruction}

Generate ONE interview question. Keep it under 30 words. Be direct and professional."""),
        ("user", "Generate the next interview question:")
    ])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

    return prompt | llm, phase_instructions


def advance_phase(current_phase: str, question_count: int) -> str:
    """Determine next interview phase based on progress.

    Args:
        current_phase: Current phase
        question_count: Number of questions asked

    Returns:
        Next phase name
    """

    # For MAX_QUESTIONS=3:
    # Q1 (count=0): intro
    # Q2 (count=1): behavioral
    # Q3 (count=2): product_sense

    if question_count == 0:
        return "intro"
    elif question_count == 1:
        return "behavioral"
    elif question_count == 2:
        return "product_sense"
    else:
        return "feedback"  # Shouldn't reach here if MAX_QUESTIONS check works


def generate_question(state: InterviewState) -> Dict[str, Any]:
    """Generate a new interview question based on current phase and context.

    Args:
        state: Current interview state

    Returns:
        State updates with new question and updated phase/count
    """

    question_count = state.get("question_count", 0)
    current_phase = state.get("current_phase", "intro")

    # Check if we've reached max questions (count starts at 0, so after 3 questions it will be 3)
    # We want to stop BEFORE asking the 4th question
    if question_count >= MAX_QUESTIONS:
        return {
            "messages": [AIMessage(content="Thank you for your time. Let's wrap up the interview.")],
            "current_phase": "feedback",
            "question_count": question_count  # Don't increment
        }

    # Advance phase if needed
    new_phase = advance_phase(current_phase, question_count)

    # Generate question
    chain, phase_instructions = create_question_generator_chain(new_phase)
    response = chain.invoke({
        "company_name": state.get("company_name", "the company"),
        "role_title": state.get("role_title", "this role"),
        "company_values": ", ".join(state.get("company_values", [])) or "innovation and excellence",
        "required_skills": ", ".join(state.get("required_skills", [])) or "problem-solving and communication",
        "phase_instruction": phase_instructions.get(new_phase, "")
    })

    new_question = response.content

    # Add conversational transition if not the first question
    # Only add transition if there are both AI and human messages (Q&A exchange happened)
    messages = state.get("messages", [])
    has_human_message = any(hasattr(msg, 'type') and msg.type == 'human' for msg in messages)

    if has_human_message:
        import random

        # Check if coming from drill-down mode
        in_drilldown = state.get("in_drilldown_mode", False)
        last_score = state.get("last_similarity_score", 0.5)

        # Only show "needs improvement" if BOTH:
        # 1. Coming from drill-down mode (second attempt)
        # 2. Answer was STILL weak (below threshold)
        from devils_advocate.config import SIMILARITY_THRESHOLD
        if in_drilldown and last_score < SIMILARITY_THRESHOLD:
            # Special transition for post-drill-down (answer was still weak)
            transitions = [
                "That still needs improvement, but let's move to the next question. ",
                "We'll come back to that later. Let's move on. ",
                "Let's continue. Next question: ",
                "Okay, moving forward. "
            ]
        else:
            # Normal transition (answer was good, or improved after drill-down)
            transitions = [
                "Okay, next question. ",
                "Great, let's move on. ",
                "Alright, moving forward. ",
                "Good. Next, "
            ]

        transition = random.choice(transitions)
        new_question = transition + new_question

    return {
        "messages": [AIMessage(content=new_question)],
        "current_phase": new_phase,
        "question_count": question_count + 1,
        "in_drilldown_mode": False  # Reset drill-down flag when generating new question
    }
