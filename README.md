# Devil's Advocate - AI Mock Interviewer

An intelligent, voice-first mock interviewer that adapts to specific job descriptions and challenges weak answers in real-time using advanced multi-agent AI architecture.

## Demo

[https://github.com/divyahegde-07/Devil-s-Advocate-AI-Mock-Interviewer/blob/1873a4e8432995234dd57009d4d40e1c28dd2ec8/demo/DevilsAdvocateDemo.mp4](https://github.com/divyahegde-07/Devil-s-Advocate-AI-Mock-Interviewer/blob/1873a4e8432995234dd57009d4d40e1c28dd2ec8/demo/DevilsAdvocateDemo.mp4)

## What Makes This Different

Unlike traditional mock interview tools, this tool doesn't just ask scripted questions. It uses a **Reflexion Pattern** where a hidden AI agent semantically evaluates every answer against ideal response patterns. When it detects vague or weak responses, it automatically enters "Drill-Down Mode" with skeptical follow-up questions—just like a real interviewer would.

### Key Features

- **Context-Aware Questions**: Paste any job description, and the system generates role-specific questions tailored to the company's values and required skills
- **Semantic Answer Evaluation**: Uses LLM-as-judge scoring (0-100) to evaluate answer quality, not just keyword matching
- **Adaptive Questioning**: Automatically probes weak answers with targeted follow-ups
- **Voice-First Design**: Real-time speech-to-text and text-to-speech with <300ms latency
- **Dynamic Voice Modulation**: AI voice becomes more expressive and skeptical when challenging weak answers
- **Detailed Scorecard**: Post-interview report with identified flaws and specific improvement suggestions

## Technical Architecture

### Multi-Agent System

The system orchestrates four specialized AI agents using **LangGraph**:

```
User Answer
    ↓
[Analyzer Agent] ──→ LLM-as-Judge Scoring (0-100)
    ↓
[Router Agent] ──→ Conditional Logic (Score < 75?)
    ↓
    ├─ Weak → [Drill-Down Agent] → Follow-up Question
    └─ Strong → [Question Generator] → Next Question
    ↓
Voice Output (ElevenLabs)
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Orchestration** | LangGraph | State machine for conversation flow |
| **LLM** | GPT-4o | Question generation, analysis, scoring |
| **Frontend** | Streamlit | Web interface with glassmorphism UI |
| **STT** | OpenAI Whisper | Speech-to-text transcription |
| **TTS** | ElevenLabs Turbo v2 | Text-to-speech (<300ms latency) |
| **Vector DB** | ChromaDB | Semantic search for ideal answer patterns |
| **Validation** | Pydantic | Type-safe data schemas |

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API Key
- ElevenLabs API Key

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Devils\ Advocate
```

2. **Install dependencies**
```bash
pip install -e .
```

Or using `uv` (recommended):
```bash
uv sync
```

3. **Set up environment variables**

Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here  
```

### Running the Application

```bash
streamlit run app.py
```

## How to Use

1. **Paste Job Description**: Copy and paste any job posting into the text area
2. **Start Interview**: The AI extracts role context and generates the first question
3. **Respond**: Use voice recording or type your answers
4. **Get Challenged**: Weak answers trigger follow-up questions automatically
5. **Review Report**: View your performance scorecard with detailed feedback

## Interview Phases

The system progresses through structured phases:

1. **Intro** (Q1): Warm-up question about background
2. **Behavioral** (Q2): STAR-format questions about past experiences
3. **Product Sense** (Q3): Product thinking and problem-solving

Default: 3 questions total (configurable via `MAX_QUESTIONS`). This can be changed based on what the user wants - applies to phases as well as number of questions.

## Architecture Deep Dive

### Reflexion Pattern Implementation

The **Analyzer Agent** doesn't just score answers—it uses a hybrid approach:

1. Retrieves top 3 relevant "ideal intents" from ChromaDB vector store
2. Passes them as context to GPT-4o for nuanced scoring
3. Sets skepticism level based on score (0.75 threshold)

This combines retrieval and generation for accurate evaluation.

### Smart Drill-Down Logic

- Only challenges weak answers **once** (prevents infinite loops)
- Uses background threading for detailed flaw analysis (non-blocking)
- Router agent prevents re-drilling on the same question

### Voice Pipeline Optimization

- ElevenLabs Turbo v2 for fastest response
- Audio caching per question (avoids regeneration)
- Dynamic stability control:
  - Normal mode: 0.5 (professional tone)
  - Drill-down mode: 0.35 (expressive, slightly impatient)

## Configuration

Key parameters in `devils_advocate/config.py`:

```python
SIMILARITY_THRESHOLD = 0.75        # Cutoff for weak answers
WEAK_ANSWER_SKEPTICISM = 0.8       # High skepticism for weak answers
STRONG_ANSWER_SKEPTICISM = 0.2     # Low skepticism for strong answers
MAX_QUESTIONS = 3                   # Interview length
LLM_MODEL = "gpt-4o"               # OpenAI model
LLM_TEMPERATURE = 0.7              # Response creativity
```

## Project Structure

```
devils_advocate/
├── agents/                 # Multi-agent system
│   ├── analyzer.py        # LLM-as-judge answer scorer
│   ├── drill_down.py      # Follow-up question generator
│   ├── question_generator.py  # Role-specific questions
│   └── router.py          # Conditional routing logic
├── chains/
│   └── context_extractor.py  # JD parsing with Pydantic
├── models/
│   ├── schemas.py         # Data models
│   └── state.py           # LangGraph state definition
├── utils/
│   ├── stt.py            # Whisper integration
│   ├── tts.py            # ElevenLabs integration
│   └── vector_store.py   # ChromaDB setup
├── config.py              # Configuration management
└── graph.py               # LangGraph assembly
```

## Technical Highlights

### Why LangGraph?
- Conditional routing based on answer quality
- Built-in conversation memory and checkpointing
- Clear graph visualization for debugging

### Why LLM-as-Judge?
- More nuanced than cosine similarity (understands context)
- Handles paraphrasing and semantic equivalence
- Provides structured 0-100 scoring with reasoning

### Why Ephemeral ChromaDB?
- No persistence needed (ideal intents are static)
- Faster initialization (no disk I/O)
- Simpler deployment (no external database)

### Performance Optimizations
- Cached embeddings instance (avoid recreation)
- Background threading for flaw analysis
- In-memory vector store
- Audio caching per question

## UI Design

Features a futuristic glassmorphism design with:
- Dark blue/black gradient background
- Animated particle effects
- Glass-morphic containers
- Smooth transitions and hover effects

## Answer Evaluation

Answers are scored 0-100 based on:
- **0-40**: Off-topic, nonsense, or gibberish
- **40-60**: Vague, lacks specifics
- **60-75**: Decent but missing key details
- **75-90**: Good with concrete examples
- **90-100**: Excellent, comprehensive, insightful

The system compares against 20+ ideal intent patterns:
- **Behavioral**: Data-driven decisions, ownership, collaboration, learning from failure
- **Product Sense**: User needs, metrics, tradeoffs, structured thinking

## Contributing

This project demonstrates advanced AI engineering patterns:
- Multi-agent orchestration
- Semantic answer evaluation
- Adaptive conversational flows
- Type-safe state management

## License

[Add your license here]

## Acknowledgments

Built with:
- [LangChain](https://langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [OpenAI](https://openai.com/) (GPT-4o & Whisper)
- [ElevenLabs](https://elevenlabs.io/) (Voice synthesis)
- [Streamlit](https://streamlit.io/) (Web framework)

---

**Note**: This is an educational project for interview preparation. API costs apply for OpenAI and ElevenLabs usage.
