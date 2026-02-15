"""Text-to-Speech integration using ElevenLabs streaming API."""

from typing import Iterator
from elevenlabs.client import ElevenLabs

from devils_advocate.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    NORMAL_STABILITY,
    DRILL_DOWN_STABILITY
)


def text_to_speech_stream(
    text: str,
    skepticism_level: float = 0.5,
    voice_id: str = None
) -> Iterator[bytes]:
    """Stream audio bytes from text using ElevenLabs.

    Args:
        text: Text to convert to speech
        skepticism_level: 0.0-1.0, controls voice expressiveness
        voice_id: Optional specific voice ID

    Yields:
        Audio byte chunks for streaming playback
    """

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Use provided voice_id or default from config
    vid = voice_id or ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel

    # Map skepticism to stability (high skepticism = low stability = more expressive)
    stability = NORMAL_STABILITY
    if skepticism_level > 0.6:
        stability = DRILL_DOWN_STABILITY

    # Stream audio using the correct API
    audio_stream = client.text_to_speech.convert(
        voice_id=vid,
        text=text,
        model_id="eleven_turbo_v2",  # Fastest model for <300ms latency
        voice_settings={
            "stability": stability,
            "similarity_boost": 0.75
        }
    )

    # Return the audio bytes (convert returns complete audio, not stream)
    yield audio_stream


def text_to_speech(
    text: str,
    skepticism_level: float = 0.5,
    voice_id: str = None
) -> bytes:
    """Convert text to speech (non-streaming).

    Args:
        text: Text to convert to speech
        skepticism_level: 0.0-1.0, controls voice expressiveness
        voice_id: Optional specific voice ID

    Returns:
        Complete audio as bytes
    """

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Use provided voice_id or default from config
    vid = voice_id or ELEVENLABS_VOICE_ID or "21m00Tcm4TlvDq8ikWAM"

    # Map skepticism to stability
    stability = NORMAL_STABILITY
    if skepticism_level > 0.6:
        stability = DRILL_DOWN_STABILITY

    # Convert returns an iterator of audio chunks
    audio_generator = client.text_to_speech.convert(
        voice_id=vid,
        text=text,
        model_id="eleven_turbo_v2",
        voice_settings={
            "stability": stability,
            "similarity_boost": 0.75
        }
    )

    # Collect all chunks and combine them
    audio_chunks = list(audio_generator)
    return b"".join(audio_chunks)
