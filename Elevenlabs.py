"""
English call transcription with speaker diarization (ElevenLabs Scribe).

Usage:
    python transcribe.py call.mp3
    python transcribe.py call.mp3 --speakers 2
    python transcribe.py call.mp3 --language eng     # language code (default: eng)

Setup:
    pip install elevenlabs python-dotenv
    ffmpeg must be installed and on PATH
    .env file containing:  ELEVENLABS_API_KEY=your_key_here
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
ELEVENLABS_MODEL = "scribe_v1"


# --------------------------------------------------------------------------
# Step 1: Audio preprocessing (single ffmpeg pass)
# --------------------------------------------------------------------------
def preprocess_audio(src: Path, dst: Path) -> None:
    """Mono, 16 kHz, band-pass filter, loudness normalise to -16 LUFS,
    trim leading silence."""
    audio_filter = ",".join([
        "highpass=f=120",                      # remove hum / rumble
        "lowpass=f=7500",                      # remove high-freq artefacts
        "loudnorm=I=-16:TP=-1.5:LRA=11",       # consistent loudness
        "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-45dB",
    ])
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
         "-af", audio_filter, str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


# --------------------------------------------------------------------------
# Step 2: ElevenLabs transcription + diarization
# --------------------------------------------------------------------------
def transcribe(client: ElevenLabs, audio: Path, num_speakers: int, language: str) -> str:
    print("  Uploading and transcribing...")
    with open(audio, "rb") as f:
        result = client.speech_to_text.convert(
            file=f,
            model_id=ELEVENLABS_MODEL,
            language_code=language,
            diarize=True,
            num_speakers=num_speakers,
            tag_audio_events=False,        # skip tags like (laughter)
        )

    # ElevenLabs returns word-by-word; group consecutive words by speaker
    # into turns: "SPEAKER_00: ..." / "SPEAKER_01: ..."
    turns: list[list] = []   # each item: [speaker_label, text]
    for w in result.words:
        if w.type == "audio_event":
            continue
        if w.type == "spacing":
            if turns:
                turns[-1][1] += w.text
            continue

        speaker = w.speaker_id or "speaker_0"
        digits = re.findall(r"\d+", speaker)
        label = f"SPEAKER_{int(digits[0]):02d}" if digits else speaker.upper()

        if not turns or turns[-1][0] != label:
            turns.append([label, w.text])
        else:
            turns[-1][1] += w.text if turns[-1][1].endswith(" ") else " " + w.text

    lines = [f"{label}: {text.strip()}" for label, text in turns if text.strip()]
    if not lines:  # fallback: no word data returned
        text = (result.text or "").strip()
        return f"SPEAKER_00: {text}" if text else ""
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="English call transcription with ElevenLabs")
    parser.add_argument("audio", type=Path, help="path to audio file")
    parser.add_argument("--speakers", type=int, default=2, help="number of speakers (default 2)")
    parser.add_argument("--language", default="eng", help="language code (default: eng)")
    args = parser.parse_args()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        sys.exit("Error: ELEVENLABS_API_KEY not set (add it to a .env file)")
    if not args.audio.exists():
        sys.exit(f"Error: file not found: {args.audio}")
    if args.audio.suffix.lower() not in AUDIO_EXTS:
        sys.exit(f"Error: unsupported format. Use one of {sorted(AUDIO_EXTS)}")

    client = ElevenLabs(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp:
        clean_audio = Path(tmp) / "clean.wav"

        print("[1/2] Preprocessing audio...")
        preprocess_audio(args.audio, clean_audio)

        print("[2/2] Transcribing with ElevenLabs...")
        transcript = transcribe(client, clean_audio, args.speakers, args.language)

    if not transcript:
        sys.exit("No speech detected.")

    out_file = args.audio.with_suffix(".transcript.txt")
    out_file.write_text(transcript, encoding="utf-8")

    print("\n" + "=" * 60)
    print(transcript)
    print("=" * 60)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()