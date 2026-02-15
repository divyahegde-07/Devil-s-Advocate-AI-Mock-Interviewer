"""DrillDown node - generates probing follow-up questions for weak answers."""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from devils_advocate.models.state import InterviewState
from devils_advocate.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def create_drill_down_chain():
    """Create chain for generating skeptical follow-up questions."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a skeptical interviewer who noticed a weakness in the candidate's answer.
Generate a SHORT, DIRECT follow-up question that probes the gap you identified.
Be professional but skeptical. Push them to provide specifics, data, or deeper reasoning.

Examples:
- "Can you walk me through the specific metrics you tracked?"
- "What would you have done differently if that approach failed?"
- "How did you validate that assumption?"

Keep it under 20 words."""),
        ("user", """Original Question: {original_question}
Their Weak Answer: {weak_answer}
Identified Flaw: {flaw}

Generate a probing follow-up question:""")
    ])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

    return prompt | llm


def drill_down(state: InterviewState) -> Dict[str, Any]:
    """Generate a follow-up question to probe a weak answer.

    Args:
        state: Current interview state with weak answer context

    Returns:
        State updates with drill-down question added to messages
    """

    # Get context from state
    messages = state.get("messages", [])
    user_answer = state.get("last_user_answer", "")
    evaluations = state.get("question_evaluations", [])

    # Get the original question and flaw
    original_question = ""
    flaw = "vague or lacking specifics"

    # Find last AI question
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            original_question = msg.content
            break

    # Get latest evaluation if available
    if evaluations:
        latest_eval = evaluations[-1]
        flaw = latest_eval.get("flaw", flaw)

    # Generate drill-down question
    chain = create_drill_down_chain()
    response = chain.invoke({
        "original_question": original_question,
        "weak_answer": user_answer,
        "flaw": flaw
    })

    drill_question = response.content

    # Add conversational transition
    transitions = [
        "Okay, let's try that one more time. ",
        "Can you elaborate on that? ",
        "Hmm, tell me more about that. ",
        "Let's dig deeper into that. "
    ]

    import random
    transition = random.choice(transitions)
    full_response = transition + drill_question

    # Add to messages and set drill-down flag
    return {
        "messages": [AIMessage(content=full_response)],
        "in_drilldown_mode": True
    }
