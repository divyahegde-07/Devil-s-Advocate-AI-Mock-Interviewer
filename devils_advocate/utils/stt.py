"""Speech-to-Text integration using OpenAI Whisper."""

import io
from openai import OpenAI

from devils_advocate.config import OPENAI_API_KEY


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio to text using OpenAI Whisper.

    Args:
        audio_bytes: Raw audio data (WAV, MP3, etc.)

    Returns:
        Transcribed text
    """

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Create a file-like object from bytes
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"  # Whisper needs a filename

    # Transcribe
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="en"  # Optional: can be omitted for auto-detection
    )

    return transcript.text


def transcribe_audio_stream(audio_chunks: list) -> str:
    """Transcribe streaming audio chunks.

    Args:
        audio_chunks: List of audio byte chunks

    Returns:
        Transcribed text
    """

    # Combine chunks into single audio file
    combined_audio = b"".join(audio_chunks)

    return transcribe_audio(combined_audio)
