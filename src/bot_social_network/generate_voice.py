#!/usr/bin/env python3

import argparse
import os
import re
import sys
import datetime
import tempfile
from google.api_core import exceptions
from google.cloud import texttospeech

# Silence the pygame community welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame


def create_filename(text, audio_format):
    """Creates a sanitized, unique filename from the input text and a timestamp."""
    sanitized_text = re.sub(r"[^\w\s-]", "", text).strip()
    sanitized_text = re.sub(r"[-\s]+", "_", sanitized_text)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Truncate to a reasonable length
    base_filename = f"{sanitized_text[:50]}_{timestamp}.{audio_format.lower()}"
    return base_filename


def get_voice_list():
    """Returns a list of available 'en-US' "Chirp3" voice names."""
    client = texttospeech.TextToSpeechClient()
    try:
        voices = client.list_voices(language_code="en-US")
        return [v.name for v in voices.voices if "Chirp3" in v.name]
    except exceptions.GoogleAPICallError as e:
        print(f"Error fetching voice list from Google Cloud: {e}", file=sys.stderr)
        print(
            "Please ensure you are authenticated. Try running 'gcloud auth application-default login'",
            file=sys.stderr,
        )
        return []


def list_voices_table(voice_list):
    """Prints the available voices in a formatted table."""
    if not voice_list:
        print("No 'en-US' 'Chirp3' voices could be fetched.", file=sys.stderr)
        return

    # To get gender, we still need the full voice objects
    client = texttospeech.TextToSpeechClient()
    voices = client.list_voices(language_code="en-US")
    chirp_voices = [v for v in voices.voices if v.name in voice_list]

    max_name_len = max(len(v.name) for v in chirp_voices) if chirp_voices else 20

    print("Available 'en-US' 'Chirp3' Voices:")
    print(f"{'Name':<{max_name_len}}  {'Gender'}")
    print(f"{'=' * max_name_len}  {'=' * 6}")

    for voice in chirp_voices:
        ssml_gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        print(f"{voice.name:<{max_name_len}}  {ssml_gender}")


def generate_voice(
    project_id: str,
    text: str,
    output_file: str,
    language_code: str,
    voice_name: str,
    audio_encoding: texttospeech.AudioEncoding,
):
    """Generates speech from text using Google Cloud Text-to-Speech."""
    print(f"Using project: '{project_id}'", file=sys.stderr)
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, name=voice_name
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=audio_encoding)

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with open(output_file, "wb") as out:
            out.write(response.audio_content)

        if "tmp" not in output_file:
            print(f'Audio content written to file "{output_file}"', file=sys.stderr)

    except exceptions.GoogleAPICallError as e:
        print(f"Error during speech synthesis: {e}", file=sys.stderr)
        print("Please check your authentication and API permissions.", file=sys.stderr)
        return False
    return True


def play_audio(file_path):
    """Plays an audio file using pygame."""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except KeyboardInterrupt:
        pygame.mixer.music.stop()
        print("\nPlayback interrupted by user.", file=sys.stderr)


def main():
    """Parses command-line arguments and calls the voice generation function."""
    parser = argparse.ArgumentParser(
        description="Generate speech from text using Google Cloud Text-to-Speech. Accepts piped input.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Input Arguments ---
    input_group = parser.add_argument_group("Input Options (provide one)")
    input_group.add_argument(
        "text",
        nargs="?",
        type=str,
        default=None,
        help="The text to synthesize. Can be omitted if using --input-file or piping data.",
    )
    input_group.add_argument(
        "--input-file", type=str, help="Path to a text file to synthesize."
    )

    # --- Output Arguments ---
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="The name of the output audio file. If not specified, a name is generated. Ignored if --temp is used.",
    )
    output_group.add_argument(
        "--audio-format",
        type=str,
        default="MP3",
        choices=["MP3", "WAV", "OGG"],
        help="The format of the output audio file. WAV is uncompressed, OGG uses the Ogg Opus codec.",
    )
    output_group.add_argument(
        "--temp",
        action="store_true",
        help="Use a temporary file for audio playback, which is deleted after playing.",
    )
    output_group.add_argument(
        "--no-play", action="store_true", help="Do not play the generated audio."
    )

    # --- Voice Configuration ---
    voice_group = parser.add_argument_group("Voice Configuration")
    voice_group.add_argument(
        "--language-code",
        type=str,
        default="en-US",
        help="The language code for the voice.",
    )
    voice_group.add_argument(
        "--voice-name",
        type=str,
        default="en-US-Chirp3-HD-Achernar",
        help="The name of the voice to use. Use --list-voices to see available options.",
    )
    voice_group.add_argument(
        "--list-voices",
        action="store_true",
        help="List available 'en-US-Chirp3' voices and exit.",
    )

    # --- Project Configuration ---
    project_group = parser.add_argument_group("Project Configuration")
    project_group.add_argument(
        "--project-id",
        type=str,
        default=os.environ.get("GCLOUD_PROJECT", "ucr-research-computing"),
        help="Your Google Cloud project ID.",
    )

    args = parser.parse_args()

    # --- Logic ---
    try:
        if args.list_voices:
            valid_voices = get_voice_list()
            list_voices_table(valid_voices)
            return

        text_to_synthesize = ""
        # Check for input in order of precedence: --input-file, then text argument, then piped data.
        if args.input_file:
            if args.text:
                parser.error("argument --input-file: not allowed with a text argument.")
            try:
                with open(args.input_file, "r") as f:
                    text_to_synthesize = f.read()
            except FileNotFoundError:
                parser.error(f"Input file not found: {args.input_file}")
        elif args.text:
            text_to_synthesize = args.text
        elif not sys.stdin.isatty():
            print("Reading from pipe (stdin)...", file=sys.stderr)
            text_to_synthesize = sys.stdin.read().strip()

        if not text_to_synthesize:
            parser.error(
                "No input provided. Provide text as an argument, use --input-file, or pipe data to the script."
            )

        if args.temp and args.no_play:
            parser.error("--temp cannot be used with --no-play.")

        valid_voices = get_voice_list()
        if valid_voices and args.voice_name not in valid_voices:
            parser.error(
                f"Invalid voice name: '{args.voice_name}'.\nUse --list-voices to see available options."
            )

        audio_encoding_map = {
            "MP3": texttospeech.AudioEncoding.MP3,
            "WAV": texttospeech.AudioEncoding.LINEAR16,
            "OGG": texttospeech.AudioEncoding.OGG_OPUS,
        }
        audio_encoding = audio_encoding_map[args.audio_format]

        common_args = {
            "project_id": args.project_id,
            "text": text_to_synthesize,
            "language_code": args.language_code,
            "voice_name": args.voice_name,
            "audio_encoding": audio_encoding,
        }

        if args.temp:
            if args.output_file:
                print(
                    "Warning: --output-file is ignored when --temp is used.",
                    file=sys.stderr,
                )

            suffix = f".{args.audio_format.lower()}"
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=True
            ) as temp_audio_file:
                temp_filename = temp_audio_file.name
                if generate_voice(**common_args, output_file=temp_filename):
                    print("Playing temporary audio file...", file=sys.stderr)
                    play_audio(temp_filename)
                    print("Temporary file will be deleted.", file=sys.stderr)
        else:
            output_filename = (
                args.output_file
                if args.output_file
                else create_filename(text_to_synthesize, args.audio_format)
            )
            if (
                generate_voice(**common_args, output_file=output_filename)
                and not args.no_play
            ):
                play_audio(output_filename)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
