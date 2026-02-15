"""Configuration management for Devils Advocate."""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Analyzer Settings
SIMILARITY_THRESHOLD = 0.75  # Threshold for weak vs strong answers (stricter)
WEAK_ANSWER_SKEPTICISM = 0.8  # High skepticism for weak answers
STRONG_ANSWER_SKEPTICISM = 0.2  # Low skepticism for strong answers

# Voice Settings
NORMAL_STABILITY = 0.5  # Voice stability for normal mode
DRILL_DOWN_STABILITY = 0.35  # More expressive/impatient for drill-down

# Interview Settings
MAX_QUESTIONS = 3  # Maximum questions before auto-ending interview
QUESTION_TIMEOUT_SECONDS = 120  # Time limit per question

# Model Settings
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.7

# ChromaDB Settings
CHROMA_COLLECTION_NAME = "ideal_intents"
