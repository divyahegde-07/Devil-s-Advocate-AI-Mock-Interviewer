"""Analyzer node - evaluates user answers using LLM-as-judge scoring."""

from typing import Dict, Any
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
import threading

from devils_advocate.models.state import InterviewState
from devils_advocate.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SIMILARITY_THRESHOLD,
    WEAK_ANSWER_SKEPTICISM,
    STRONG_ANSWER_SKEPTICISM
)

# Cache embeddings instance to avoid recreation overhead
_embeddings_cache = None
# Thread pool for background flaw analysis
_executor = ThreadPoolExecutor(max_workers=2)
# Lock for thread-safe evaluation storage
_eval_lock = threading.Lock()

def get_embeddings():
    """Get or create cached embeddings instance."""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return _embeddings_cache


def create_analyzer_chain():
    """Create LLM chain for identifying flaws in weak answers."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a critical interviewer evaluating answers.
Identify the KEY FLAW in this answer and provide a specific suggestion for improvement.
Be concise and direct."""),
        ("user", """Question: {question}
User's Answer: {answer}

Identify the main weakness and how to improve it.""")
    ])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

    return prompt | llm


def create_scoring_chain():
    """Create LLM chain for scoring answer quality (0-100)."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert interviewer evaluating answer quality.

Rate the answer from 0-100 based on:
- 0-40: Nonsense, no substance, off-topic, or gibberish
- 40-60: Vague, lacks specifics, or minimal effort
- 60-75: Decent but missing key details or examples
- 75-90: Good answer with concrete examples and reasoning
- 90-100: Excellent, comprehensive, insightful

Consider these ideal characteristics:
{ideal_intents}

Return ONLY a number from 0-100. No explanation."""),
        ("user", """Question: {question}
User's Answer: {answer}

Score (0-100):""")
    ])

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,  # Lower temperature for consistent scoring
        api_key=OPENAI_API_KEY
    )

    return prompt | llm


def analyze_answer(state: InterviewState, vector_store) -> Dict[str, Any]:
    """Analyze user's answer using semantic similarity.

    This node:
    1. Vectorizes the user's answer
    2. Compares against ideal intents in ChromaDB
    3. Sets skepticism level based on similarity
    4. If weak, generates flaw analysis

    Args:
        state: Current interview state
        vector_store: ChromaDB vector store with ideal intents

    Returns:
        State updates with similarity score and skepticism
    """

    user_answer = state.get("last_user_answer", "")
    if not user_answer:
        return {}

    # Get the last question from messages
    messages = state.get("messages", [])
    last_question = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            last_question = msg.content
            break

    # Search for relevant ideal intents (for context)
    results = vector_store.similarity_search_with_score(
        user_answer,
        k=3  # Get top 3 intents for context
    )

    # Format ideal intents for LLM judge
    ideal_intents = ""
    if results:
        ideal_intents = "\n".join([f"- {doc.page_content}" for doc, _ in results])
    else:
        ideal_intents = "- Clear structure and reasoning\n- Specific examples\n- Concrete details"

    # Use LLM to score answer quality (0-100)
    scoring_chain = create_scoring_chain()
    try:
        score_response = scoring_chain.invoke({
            "question": last_question,
            "answer": user_answer,
            "ideal_intents": ideal_intents
        })

        # Extract numeric score from response
        score_text = score_response.content.strip()
        # Handle responses like "85", "Score: 85", or "85/100"
        match = re.search(r'\d+', score_text)
        if match:
            llm_score = int(match.group())
            # Convert 0-100 to 0-1 range
            similarity_score = llm_score / 100.0
        else:
            similarity_score = 0.5  # Fallback if parsing fails
    except Exception:
        # Fallback to neutral score if LLM scoring fails
        similarity_score = 0.5

    # Determine if answer is weak or strong
    is_weak = similarity_score < SIMILARITY_THRESHOLD

    # Set skepticism level
    skepticism = WEAK_ANSWER_SKEPTICISM if is_weak else STRONG_ANSWER_SKEPTICISM

    updates = {
        "last_similarity_score": float(similarity_score),
        "skepticism_level": float(skepticism)
    }

    # Get the last question from messages
    messages = state.get("messages", [])
    last_question = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            last_question = msg.content
            break

    # Create evaluation for ALL answers (weak and strong)
    evaluation = {
        "question": last_question,
        "user_answer": user_answer,
        "score": float(similarity_score),
        "flaw": "Good answer" if not is_weak else "Analyzing...",
        "suggestion": "Keep up the quality!" if not is_weak else "Analyzing..."
    }

    current_evals = state.get("question_evaluations", [])
    updates["question_evaluations"] = current_evals + [evaluation]

    # If weak answer, run detailed flaw analysis in background thread
    # This provides quality feedback without blocking the conversation
    if is_weak:
        # Run detailed flaw analysis in background (non-blocking)
        def analyze_flaw_background():
            """Background thread to analyze flaw with LLM."""
            try:
                analyzer_chain = create_analyzer_chain()
                flaw_response = analyzer_chain.invoke({
                    "question": last_question,
                    "answer": user_answer
                })
                flaw_text = flaw_response.content

                # Update evaluation in state (thread-safe)
                # Note: This update happens async, so it might not be in current state
                # but will be available for final report
                evaluation["flaw"] = flaw_text
                evaluation["suggestion"] = flaw_text
            except Exception as e:
                # Fallback to generic if analysis fails
                evaluation["flaw"] = f"Could not analyze: {str(e)}"
                evaluation["suggestion"] = "Provide more details and specific examples"

        # Submit to thread pool (returns immediately)
        _executor.submit(analyze_flaw_background)

    return updates
