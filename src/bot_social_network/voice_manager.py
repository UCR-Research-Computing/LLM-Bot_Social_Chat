import os
import sys
import logging
from typing import List, Dict, Optional

from google.api_core import exceptions
from google.cloud import texttospeech

# Silence the pygame community welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

# --- Globals ---
_voice_cache: Dict[str, List[str]] = {"male": [], "female": []}

# --- Core Functions ---


def get_voices() -> Dict[str, List[str]]:
    """
    Fetches and caches a list of available 'en-US' "Chirp3" voices, categorized by gender.
    Returns the cached list if already populated.
    """
    if _voice_cache.get("male") or _voice_cache.get("female"):
        return _voice_cache

    print("Fetching available voices from Google Cloud...", file=sys.stderr)
    try:
        client = texttospeech.TextToSpeechClient()
        voices_response = client.list_voices(language_code="en-US")

        for voice in voices_response.voices:
            if "Chirp3" in voice.name:
                gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name.lower()
                if gender in _voice_cache:
                    _voice_cache[gender].append(voice.name)

        logging.info(
            f"Loaded {_voice_cache['male']} male and {_voice_cache['female']} female voices."
        )
        return _voice_cache

    except exceptions.GoogleAPICallError as e:
        logging.error(f"Error fetching voice list from Google Cloud: {e}")
        logging.error(
            "Please ensure you are authenticated. Try running 'gcloud auth application-default login'"
        )
        return {"male": [], "female": []}


def select_voice(bot_name: str) -> Optional[str]:
    """Selects a voice for a bot based on a hash of its name."""
    if not _voice_cache["male"] and not _voice_cache["female"]:
        logging.warning("No voices loaded, cannot select a voice.")
        return None

    # Determine gender based on hash of bot name
    if hash(bot_name) % 2 == 0:
        gender_voices = _voice_cache["female"]
        if not gender_voices:  # Fallback to male if no female voices
            gender_voices = _voice_cache["male"]
    else:
        gender_voices = _voice_cache["male"]
        if not gender_voices:  # Fallback to female if no male voices
            gender_voices = _voice_cache["female"]

    if not gender_voices:
        return None

    # Select voice from the chosen gender list
    voice_index = hash(bot_name) % len(gender_voices)
    return gender_voices[voice_index]


async def generate_voice_file(text: str, voice_name: str, output_path: str) -> bool:
    """Generates speech from text and saves it to a file."""
    logging.info(
        f"Requesting TTS for text: '{text[:50]}...'",
        extra={"event": "tts.generate.start", "voice_name": voice_name},
    )
    try:
        client = texttospeech.TextToSpeechAsyncClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = await client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_path, "wb") as out:
            out.write(response.audio_content)

        logging.info(
            f"Successfully saved TTS audio to {output_path}",
            extra={"event": "tts.generate.success"},
        )
        return True

    except exceptions.GoogleAPICallError as e:
        logging.error(
            f"Error during speech synthesis: {e}", extra={"event": "tts.generate.fail"}
        )
    except Exception as e:
        logging.error(
            f"An unexpected error occurred in voice generation: {e}",
            extra={"event": "tts.generate.fail"},
        )
    return False


def play_audio_file(file_path: str):
    """Plays an audio file using pygame."""
    try:
        if not os.path.exists(file_path):
            logging.error(
                f"Audio file not found for playback: {file_path}",
                extra={"event": "tts.playback.fail"},
            )
            return

        logging.info(
            f"Starting audio playback for {file_path}",
            extra={"event": "tts.playback.start"},
        )
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        logging.info("Audio playback finished.", extra={"event": "tts.playback.finish"})

    except Exception as e:
        logging.error(f"Error playing audio: {e}", extra={"event": "tts.playback.fail"})
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.quit()


def stop_audio():
    """Stops any currently playing audio."""
    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        pygame.mixer.quit()
