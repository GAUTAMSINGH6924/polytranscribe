"""Route a call recording to ElevenLabs (English) or Sarvam (Hindi/Punjabi).

Usage:
    python transcribe_auto.py call.mp3
    python transcribe_auto.py call.mp3 --speakers 2 --no-llm

Setup:
    pip install elevenlabs sarvamai python-dotenv
    ffmpeg must be installed and on PATH
    .env file containing SARVAM_API_KEY and ELEVENLABS_API_KEY
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from sarvamai import SarvamAI

from Elevenlabs import AUDIO_EXTS, preprocess_audio, transcribe as transcribe_english
from Sarvam import format_transcript, llm_correct, transcribe as transcribe_sarvam

load_dotenv()


def sample_transcript_lines(transcript: str, sample_size: int = 5) -> dict[str, list[str]]:
    """Return up to five non-empty transcript lines from the start, middle, and end."""
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if not lines:
        return {"start": [], "middle": [], "end": []}

    middle_start = max(0, (len(lines) - sample_size) // 2)
    return {
        "start": lines[:sample_size],
        "middle": lines[middle_start:middle_start + sample_size],
        "end": lines[-sample_size:],
    }


def detect_language(transcript: str) -> tuple[str, str]:
    """Route to Sarvam when sampled lines contain Devanagari or Gurmukhi text."""
    samples = sample_transcript_lines(transcript)
    has_hindi = False
    has_punjabi = False
    for lines in samples.values():
        for line in lines:
            speech = re.sub(r"^[^:]+:\s*", "", line)
            for char in speech:
                codepoint = ord(char)
                if 0x0A00 <= codepoint <= 0x0A7F:
                    has_punjabi = True
                elif 0x0900 <= codepoint <= 0x097F:
                    has_hindi = True
    if has_hindi and has_punjabi:
        return "sarvam", "mixed"
    if has_punjabi:
        return "sarvam", "pa-IN"
    if has_hindi:
        return "sarvam", "hi-IN"
    return "elevenlabs", "eng"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically route English, Hindi, or Punjabi call transcription"
    )
    parser.add_argument("audio", type=Path, help="path to audio file")
    parser.add_argument("--speakers", type=int, default=2, help="number of speakers (default 2)")
    parser.add_argument("--no-llm", action="store_true", help="skip Sarvam transcript correction")
    args = parser.parse_args()

    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_key:
        sys.exit("Error: SARVAM_API_KEY not set (add it to a .env file)")
    if not args.audio.exists():
        sys.exit(f"Error: file not found: {args.audio}")
    if args.audio.suffix.lower() not in AUDIO_EXTS:
        sys.exit(f"Error: unsupported format. Use one of {sorted(AUDIO_EXTS)}")

    sarvam_client = SarvamAI(api_subscription_key=sarvam_key)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clean_audio = tmp_dir / "clean.wav"

        print("[1/3] Preprocessing audio...")
        preprocess_audio(args.audio, clean_audio)

        print("[2/3] Creating multilingual draft with Sarvam...")
        draft_dir = tmp_dir / "draft"
        draft_dir.mkdir()
        draft_result = transcribe_sarvam(
            sarvam_client,
            clean_audio,
            draft_dir,
            args.speakers,
            language_code="unknown",
        )
        draft = format_transcript(draft_result)

        if not draft:
            sys.exit("No speech detected.")

        provider, language = detect_language(draft)
        print(f"  Detected route: {language} ({provider})")

        if provider == "elevenlabs":
            elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
            if not elevenlabs_key:
                sys.exit("Error: ELEVENLABS_API_KEY not set (add it to a .env file)")
            client = ElevenLabs(api_key=elevenlabs_key)
            print("[3/3] Transcribing with ElevenLabs...")
            final = transcribe_english(client, clean_audio, args.speakers, language)
        else:
            final = draft
            if language == "pa-IN" and not args.no_llm:
                print("[3/3] Correcting Punjabi transcript with Sarvam LLM...")
                final = llm_correct(sarvam_client, draft)
            else:
                print("[3/3] Keeping Sarvam draft (no language-specific correction applied).")

    if not final:
        sys.exit("No speech detected.")

    out_file = args.audio.with_suffix(".transcript.txt")
    out_file.write_text(final, encoding="utf-8")

    print("\n" + "=" * 60)
    print(final)
    print("=" * 60)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
