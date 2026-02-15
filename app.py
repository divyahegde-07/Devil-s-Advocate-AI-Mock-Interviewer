"""Main Streamlit application for Devils Advocate interview system."""

import streamlit as st
from langchain_core.messages import HumanMessage
import base64
import io
from audio_recorder_streamlit import audio_recorder

from devils_advocate.chains.context_extractor import extract_context
from devils_advocate.graph import create_interview_graph, initialize_interview_state
from devils_advocate.utils.vector_store import create_vector_store
from devils_advocate.utils.stt import transcribe_audio
from devils_advocate.utils.tts import text_to_speech


# Page config
st.set_page_config(
    page_title="Devil's Advocate - AI Interviewer",
    page_icon="🎙️",
    layout="wide"
)

# Futuristic CSS styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    /* CSS Variables for theming - Dark Blue/Black */
    :root {
        --bg-primary: #000000;
        --bg-secondary: #0a0e1a;
        --accent-primary: #4a5f8f;
        --accent-secondary: #6b7fa8;
        --text-primary: #e8eaed;
        --text-secondary: #9aa0a6;
        --glass-bg: rgba(10, 14, 26, 0.7);
        --glass-border: rgba(74, 95, 143, 0.2);
    }

    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0a0e1a 50%, #050810 100%);
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Subtle background particles - much dimmer */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            radial-gradient(1px 1px at 20% 30%, rgba(212, 175, 55, 0.05), transparent),
            radial-gradient(1px 1px at 60% 70%, rgba(251, 191, 36, 0.05), transparent),
            radial-gradient(1px 1px at 50% 50%, rgba(212, 175, 55, 0.03), transparent);
        background-size: 200% 200%;
        background-position: 0% 0%;
        animation: particle-float 30s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes particle-float {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
    }

    /* Glass morphism containers */
    .element-container, .stMarkdown, .stTextArea, .stButton {
        z-index: 1;
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        text-shadow: 0 2px 8px rgba(212, 175, 55, 0.1);
    }

    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Text colors */
    p, .stMarkdown {
        color: var(--text-secondary) !important;
    }

    /* Glass morphism cards */
    .stTextArea > div > div,
    [data-testid="stExpander"],
    .stMetric {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: #1a1d23 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(212, 175, 55, 0.2) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(212, 175, 55, 0.3) !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-secondary), var(--accent-primary)) !important;
        box-shadow: 0 4px 16px rgba(251, 191, 36, 0.3) !important;
    }

    /* Input fields */
    .stTextArea textarea, .stTextInput input {
        background: rgba(26, 29, 35, 0.8) !important;
        border: 1px solid rgba(212, 175, 55, 0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 1rem !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.2) !important;
    }

    /* Metrics */
    .stMetric {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        padding: 1.5rem !important;
        border-radius: 16px !important;
        border: 1px solid var(--glass-border) !important;
    }

    .stMetric label {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: var(--accent-primary) !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        text-shadow: 0 2px 8px rgba(212, 175, 55, 0.2);
    }

    /* Info/Success/Warning boxes */
    .stInfo, .stSuccess, .stWarning {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border-left: 4px solid var(--accent-primary) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
    }

    [data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-primary) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(0, 217, 255, 0.2) !important;
        margin: 2rem 0 !important;
    }

    /* Columns */
    .stColumn {
        padding: 0.5rem !important;
    }

    /* Pulse animation for active elements */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 8px rgba(212, 175, 55, 0.2); }
        50% { box-shadow: 0 0 12px rgba(212, 175, 55, 0.3); }
    }

    /* Slide-in animation */
    @keyframes slide-in {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .element-container {
        animation: slide-in 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state."""
    if "page" not in st.session_state:
        st.session_state.page = "jd_input"
    if "interview_state" not in st.session_state:
        st.session_state.interview_state = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "interview_session_1"
    if "waiting_for_answer" not in st.session_state:
        st.session_state.waiting_for_answer = False
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""


def page_jd_input():
    """Page 1: Job Description input."""
    st.title("Devil's Advocate - AI Mock Interviewer")
    st.markdown("""
    Welcome! I'll conduct a realistic mock interview based on a specific job description.
    I'll adapt my questions to the role and push back on weak answers to help you improve.
    """)

    st.subheader("Step 1: Paste the Job Description")

    jd_text = st.text_area(
        "Job Description",
       # placeholder="Paste the full job description here.",
        height=300
    )

    if st.button("Start Interview", type="primary", disabled=not jd_text):
        with st.spinner("Analyzing job description"):
            try:
                # Extract context from JD
                job_context = extract_context(jd_text)

                # Initialize vector store
                vector_store = create_vector_store()

                # Initialize interview state
                interview_state = initialize_interview_state(job_context)

                # Create graph
                graph = create_interview_graph(vector_store)

                # Store in session state
                st.session_state.interview_state = interview_state
                st.session_state.vector_store = vector_store
                st.session_state.graph = graph

                # Generate first question manually
                from devils_advocate.agents.question_generator import generate_question
                first_q = generate_question(interview_state)
                interview_state.update(first_q)
                st.session_state.interview_state = interview_state
                st.session_state.waiting_for_answer = True

                st.session_state.page = "interview"
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please check your .env file has valid API keys.")


def page_interview():
    """Page 2: Voice interview with real-time interaction."""
    st.title("🎙️ Interview in Progress")

    state = st.session_state.interview_state

    if not state:
        st.error("No interview state found. Please start over.")
        if st.button("Back to Start"):
            st.session_state.page = "jd_input"
            st.rerun()
        return

    # Display interview context
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Role: {state['role_title']} at {state['company_name']}")

    with col2:
        st.metric("Questions Asked", state.get('question_count', 0))

    # Current phase indicator
    phase_emoji = {
        "intro": "👋",
        "behavioral": "🎭",
        "product_sense": "🎯",
        "feedback": "📊"
    }
    current_phase = state.get('current_phase', 'intro')
    st.info(f"{phase_emoji.get(current_phase, '💬')} Phase: {current_phase.replace('_', ' ').title()}")

    # Display current question prominently
    messages = state.get('messages', [])
    if messages:
        # Get the last AI message (current question)
        last_ai_msg = None
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                last_ai_msg = msg.content
                break

        if last_ai_msg:
            st.markdown("### 💬 Current Question")
            st.markdown(f"**{last_ai_msg}**")
            st.markdown("---")

    # Full transcript in collapsible section
    with st.expander("📜 Show Full Transcript", expanded=False):
        messages = state.get('messages', [])
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == 'ai':
                    st.markdown(f"**🤖 Interviewer:** {msg.content}")
                elif msg.type == 'human':
                    st.markdown(f"**👤 You:** {msg.content}")

    # Play latest question as audio if available
    messages = state.get('messages', [])
    if messages:
        last_ai_msg = None
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                last_ai_msg = msg.content
                break

        if last_ai_msg:
            # Use question count as cache key to force regeneration for new questions
            question_count = state.get('question_count', 0)
            cache_key = f"audio_{question_count}_{hash(last_ai_msg)}"

            # Only generate if not cached for this specific question
            if cache_key not in st.session_state:
                try:
                    # Generate audio
                    skepticism = state.get('skepticism_level', 0.5)
                    audio_bytes = text_to_speech(last_ai_msg, skepticism)
                    st.session_state[cache_key] = audio_bytes
                except Exception as e:
                    st.warning(f"Could not generate audio: {str(e)}")
                    st.session_state[cache_key] = None

            # Play cached audio (hidden player with autoplay)
            if st.session_state.get(cache_key):
                audio_b64 = base64.b64encode(st.session_state[cache_key]).decode()
                # Use unique key for audio element to force reload
                audio_html = f'<audio key="{cache_key}" autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
                st.markdown(audio_html, unsafe_allow_html=True)

    # Microphone input for user answers
    st.subheader("🎤 Your Response")

    # Only show input if waiting for answer
    if st.session_state.waiting_for_answer:

        # Audio recorder with better visual feedback
        st.markdown("**🎤 Record your answer (click the button to start/stop):**")

        # Use question_count as part of key to reset recorder for each question
        question_count = state.get('question_count', 0)
        audio_data = audio_recorder(
            pause_threshold=2.0,  # Stop recording after 2 seconds of silence
            sample_rate=16000,
            recording_color="#ff0000",  # Bright red when recording
            neutral_color="#1f77b4",  # Blue when not recording
            icon_name="microphone",
            icon_size="3x",
            key=f"audio_recorder_{question_count}"  # Unique key per question
        )

        # If audio was recorded, transcribe it and auto-submit
        if audio_data and audio_data != st.session_state.get(f"last_audio_data_{question_count}"):
            st.session_state[f"last_audio_data_{question_count}"] = audio_data
            with st.spinner("🎧 Transcribing your answer..."):
                try:
                    transcribed = transcribe_audio(audio_data)
                    st.session_state.transcribed_text = transcribed

                    # Auto-submit after transcription
                    with st.spinner("Analyzing your answer..."):
                        # Get current state
                        current_state = st.session_state.interview_state

                        # Add user's answer to state
                        current_state["messages"].append(HumanMessage(content=transcribed))
                        current_state["last_user_answer"] = transcribed
                        current_state["transcript"] += f"\nYou: {transcribed}\n"

                        # Invoke graph to process answer and generate next question
                        graph = st.session_state.graph
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        result = graph.invoke(current_state, config)

                        # Update state
                        st.session_state.interview_state = result

                        # Clear transcribed text for next question
                        st.session_state.transcribed_text = ""

                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Text fallback with manual submit
        st.markdown("**⌨️ Or type your answer:**")
        text_input = st.text_area("Type here:", key="user_answer_text", height=100)

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Submit Typed Answer", type="primary", disabled=not text_input):
                with st.spinner("Analyzing your answer..."):
                    try:
                        # Get current state
                        current_state = st.session_state.interview_state

                        # Add user's answer to state
                        current_state["messages"].append(HumanMessage(content=text_input))
                        current_state["last_user_answer"] = text_input
                        current_state["transcript"] += f"\nYou: {text_input}\n"

                        # Invoke graph to process answer and generate next question
                        graph = st.session_state.graph
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        result = graph.invoke(current_state, config)

                        # Update state
                        st.session_state.interview_state = result

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing answer: {str(e)}")

        with col2:
            if st.button("End Interview"):
                st.session_state.page = "report"
                st.rerun()
    else:
        st.info("Generating next question...")


def page_report():
    """Page 3: Final scorecard report."""
    st.title("📊 Interview Report")

    state = st.session_state.interview_state

    if not state:
        st.error("No interview data found.")
        return

    st.subheader(f"{state['role_title']} at {state['company_name']}")

    evaluations = state.get('question_evaluations', [])

    if not evaluations:
        st.warning("No evaluations recorded yet.")
    else:
        st.markdown("### Performance Summary")

        # Create table
        for i, eval_data in enumerate(evaluations, 1):
            with st.expander(f"Question {i}: {eval_data.get('question', 'N/A')[:50]}..."):
                st.markdown(f"**Your Answer:** {eval_data.get('user_answer', 'N/A')}")
                st.markdown(f"**Score:** {eval_data.get('score', 0):.2f}")
                st.markdown(f"**Key Flaw:** {eval_data.get('flaw', 'N/A')}")
                st.markdown(f"**Suggestion:** {eval_data.get('suggestion', 'N/A')}")

    if st.button("Start New Interview"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()

    # Route to correct page
    page = st.session_state.page

    if page == "jd_input":
        page_jd_input()
    elif page == "interview":
        page_interview()
    elif page == "report":
        page_report()


if __name__ == "__main__":
    main()
