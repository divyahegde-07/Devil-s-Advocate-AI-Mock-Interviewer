# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project: Devils Advocate
- **Stack**: Python 3.11+, Streamlit, LangGraph, LangChain, OpenAI GPT-4o, ElevenLabs, Whisper, ChromaDB
- **Purpose**: A real-time, voice-first mock interviewer that adapts to Job Descriptions and pushes back on weak answers using a Reflexion Pattern.
- **Entry Point**: `app.py` - Streamlit application with 3-page flow (JD Input → Interview → Report)

## 0. Architecture Overview

### LangGraph State Machine Flow
The interview system uses a cyclic graph (`devils_advocate/graph.py`):
```
User Answer → Analyzer → Router → [Weak: DrillDown | Strong: QuestionGenerator] → END
```

**Key Nodes:**
- `analyzer` (Entry Point): Compares user answer to ideal intents via ChromaDB embeddings. Sets `similarity_score` and `skepticism_level`.
- `router`: Conditional edge. If `similarity < 0.7` → drill_down, else → question_generator.
- `drill_down`: Generates critical follow-up based on detected flaw.
- `question_generator`: Moves to next interview question.

**State Model:** `InterviewState` (TypedDict) in `devils_advocate/models/state.py` tracks:
- Job context (role_title, company_name, company_values, required_skills)
- Conversation (messages: List[BaseMessage], transcript: str)
- Logic control (current_phase, skepticism_level, question_count)
- Evaluations (question_evaluations: List[Dict], last_similarity_score)

### Key Components
1. **Context Extraction** (`devils_advocate/chains/context_extractor.py`): LangChain + Pydantic parser extracts JobContext from raw JD.
2. **Semantic Analysis** (`devils_advocate/agents/analyzer.py`): Uses OpenAIEmbeddings + ChromaDB to compute semantic similarity.
3. **Voice Pipeline**:
   - STT: `devils_advocate/utils/stt.py` (OpenAI Whisper)
   - TTS: `devils_advocate/utils/tts.py` (ElevenLabs API with adaptive voice stability)
4. **Vector Store** (`devils_advocate/utils/vector_store.py`): ChromaDB for ideal intent embeddings.

### Configuration
All settings in `devils_advocate/config.py`:
- API keys loaded from `.env` (see `.env.example` for required keys: OPENAI_API_KEY, ELEVENLABS_API_KEY)
- SIMILARITY_THRESHOLD = 0.7 (weak vs strong answer cutoff)
- WEAK_ANSWER_SKEPTICISM = 0.8, STRONG_ANSWER_SKEPTICISM = 0.2
- MAX_QUESTIONS = 3
- LLM_MODEL = "gpt-4o"

## 1. Code Style & Standards
- **General**: Write concise, idiomatic Python code. Avoid over-engineering. DRY principles apply.
- **Naming**: Use snake_case for variables/functions, PascalCase for classes.
- **Typing**: Use strict type hinting with specific types. Leverage Pydantic for data validation (see `InterviewState` in PRD).
- **Comments**: Only comment complex logic. Do NOT comment obvious code. Use docstrings for public APIs.
- **Error Handling**: Fail fast and loudly. Use custom exceptions where specific handling is needed.

## 2. Development Workflow

### Environment Setup
1. Create `.env` file from `.env.example` with valid API keys
2. Install dependencies: `uv sync`
3. Verify installation: `python -c "import devils_advocate"`

### Running the Application
- **Start Streamlit**: `streamlit run app.py`
- **Dev Mode (auto-reload)**: `streamlit run app.py --server.runOnSave true`
- **Access**: Opens browser at `http://localhost:8501`

### Testing
- No test suite currently exists (`tests/` directory is empty)
- When adding tests: Use `pytest` and mirror `devils_advocate/` structure

### Agent Behavior Rules
- **Proactiveness**: If a file is missing or path is wrong, check the file tree before asking.
- **No Hallucinations**: Do not assume libraries are installed. Check `pyproject.toml` dependencies first.
- **Brevity**: In chat, be concise. Don't explain code unless asked. Just show the diff or result.

## 3. Project Structure
```
devils_advocate/
├── agents/          # LangGraph nodes
│   ├── analyzer.py       # Semantic similarity analysis (Entry node)
│   ├── router.py         # Routing logic (weak vs strong answers)
│   ├── drill_down.py     # Critical follow-up generator
│   └── question_generator.py  # New question generator
├── chains/          # LangChain chains
│   └── context_extractor.py  # JD → JobContext parser
├── models/          # Data models
│   ├── schemas.py        # Pydantic models (JobContext)
│   └── state.py          # InterviewState TypedDict
├── utils/           # Support utilities
│   ├── stt.py            # Speech-to-text (Whisper)
│   ├── tts.py            # Text-to-speech (ElevenLabs)
│   └── vector_store.py   # ChromaDB initialization
├── config.py        # Central configuration
└── graph.py         # LangGraph assembly & state initialization
app.py               # Streamlit UI (3 pages: JD input, interview, report)
.llm/PRD.md          # Product requirements & system design
```

## 4. Critical Constraints (DO NOT IGNORE)
- **Deps**: Never introduce new dependencies without explicit permission.
- **Secrets**: NEVER output or commit API keys (OpenAI, ElevenLabs), passwords, or `.env` contents.
- **Legacy**: Do not refactor generic utility files unless specifically requested.
- **Imports**: Use absolute imports (e.g., `devils_advocate.utils`) instead of relative (`../../utils`).
- **Latency**: Voice generation must maintain <300ms latency target. Optimize streaming implementations.
