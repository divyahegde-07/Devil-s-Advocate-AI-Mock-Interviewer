# PRD: "The Devil's Advocate" — Context-Aware AI Interviewer

## 1. Project Overview
* **Goal:** Build a real-time, voice-first mock interviewer that adapts to a specific Job Description (JD) and pushes back on weak answers.
* **Core Logic:** Uses a **Reflexion Pattern**. The agent doesn't just chat; it uses a hidden "Analyzer" node to semantically compare user answers against "Ideal Intents." If a gap is detected, it enters a "Drill Down" mode.
* **User Experience:** User pastes a JD -> Voice Interview begins -> Agent adapts tone (Supportive vs. Skeptical) -> Final Scorecard Report.

## 2. Tech Stack
* **Frontend:** `Streamlit` (UI), `streamlit-webrtc` (Audio I/O).
* **Orchestration:** `LangGraph` (State Machine).
* **Voice Output:** `ElevenLabs` API (WebSocket/Streaming) for <300ms latency.
* **Voice Input:** `OpenAI Whisper` (STT).
* **LLM:** `GPT-4o` (via LangChain).
* **Memory/Eval:** `ChromaDB` (Ephemeral) for semantic intent comparison.

## 3. System Architecture & Data Flow

### Phase 1: Context Injection (The Setup)
1.  **Input:** User pastes raw JD text.
2.  **Processing:** `ContextExtractor` chain (LangChain + Pydantic) extracts:
    * `Role`
    * `Company`
    * `Values` (e.g., "Bias for Action")
    * `Required_Skills`
3.  **State Init:** `InterviewState` initialized. Vector Store loaded with "Ideal Answer" embeddings for the role.

### Phase 2: The Conversation Loop (Cyclic Graph)
1.  **Audio Capture:** `streamlit-webrtc` captures mic -> `Whisper` converts to text.
2.  **Node 1: The Analyzer (The Critic)**
    * **Action:** Vectorizes User Answer. Calculates Cosine Similarity vs. "Ideal Intent."
    * **Logic:**
        * If `Similarity < 0.7`: Mark as "Weak." Set `skepticism_level = High`.
        * If `Similarity > 0.7`: Mark as "Strong." Set `skepticism_level = Low`.
3.  **Node 2: The Router**
    * **Conditional Edge:**
        * If `Weak` -> Go to **DrillDown Node**.
        * If `Strong` -> Go to **QuestionGenerator Node** (New Topic).
4.  **Node 3: Voice Generation**
    * **Action:** LLM generates text response.
    * **Tone Control:**
        * *Normal Mode:* Stability = 0.5.
        * *Drill Down Mode:* Stability = 0.35 (More expressive/impatient).
    * **Output:** Stream audio bytes to frontend.

### Phase 3: The Report
1.  **Trigger:** "End Interview" button or Question Limit reached.
2.  **Output:** Markdown table displaying:
    * Question Asked
    * User's Key Flaw (from Analyzer)
    * Better Approach (Generated Suggestion)

## 4. State Management (`InterviewState`)

```python
from typing import TypedDict, List, Dict, Any, Annotated
import operator
from langchain_core.messages import BaseMessage

class InterviewState(TypedDict):
    # Context
    role_title: str
    company_name: str
    company_values: List[str]
    
    # Conversation
    messages: Annotated[List[BaseMessage], operator.add]
    transcript: str
    
    # Logic Control
    current_phase: str  # 'intro', 'behavioral', 'product_sense', 'feedback'
    skepticism_level: float  # 0.0 to 1.0
    
    # Scoring
    question_evaluations: List[Dict[str, Any]] 
    # { "question": str, "flaw": str, "score": int, "suggestion": str }