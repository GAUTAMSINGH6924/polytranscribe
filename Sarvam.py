import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

SARVAM_MODEL = "saaras:v3"
SARVAM_LANGUAGE = "pa-IN"
SARVAM_MODE = "codemix"          # Punjabi + Hindi/English mixed speech
LLM_CHUNK_LINES = 60             # lines per LLM correction request
LLM_MIN_RATIO = 0.80             # reject LLM output shorter than 80% of input
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

LLM_SYSTEM_PROMPT = """\
You are a Punjabi language expert correcting ASR (speech recognition) errors.

Common errors to fix:
1. Missing bindi/tippi: "ਮੈ" -> "ਮੈਂ", "ਨਹੀ" -> "ਨਹੀਂ", "ਤੁਸੀ" -> "ਤੁਸੀਂ"
2. Similar-sounding letter swaps: ਤ/ਟ, ਦ/ਡ, ਸ/ਸ਼
3. Missing nukta: ਖ਼, ਗ਼, ਜ਼, ਫ਼

Rules:
- Fix ONLY clear errors. Do not rephrase, summarize or change meaning.
- Keep every speaker label (SPEAKER_00:, SPEAKER_01:) exactly as given.
- Keep one speaker turn per line. Do not add, remove or reorder turns.
- If unsure, leave the word as-is.
- Return ONLY the corrected transcript, no explanations.\
"""


# --------------------------------------------------------------------------
# Step 1: Audio preprocessing (single ffmpeg pass)
# --------------------------------------------------------------------------
def preprocess_audio(src: Path, dst: Path) -> None:
    """Mono, 16 kHz, band-pass filter, loudness normalise to -16 LUFS,
    trim leading silence."""
    audio_filter = ",".join([
        "highpass=f=120",                      # remove hum / rumble
        "lowpass=f=7500",                      # remove high-freq artefacts
        "loudnorm=I=-16:TP=-1.5:LRA=11",       # telephony-friendly loudness
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
# Step 2: Sarvam transcription + diarization
# --------------------------------------------------------------------------
def transcribe(
    client: SarvamAI,
    audio: Path,
    out_dir: Path,
    num_speakers: int,
    language_code: str = SARVAM_LANGUAGE,
    mode: str = SARVAM_MODE,
) -> dict:
    job = client.speech_to_text_job.create_job(
        model=SARVAM_MODEL,
        mode=mode,
        language_code=language_code,
        with_diarization=True,
        num_speakers=num_speakers,
    )
    print("  Uploading audio...")
    job.upload_files(file_paths=[str(audio)])
    job.start()
    print("  Transcribing (this can take a minute)...")
    job.wait_until_complete()
    job.download_outputs(output_dir=str(out_dir))

    json_files = list(out_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError("Sarvam returned no output")
    with open(json_files[0], encoding="utf-8") as f:
        return json.load(f)


def format_transcript(result: dict) -> str:
    """Turn Sarvam's diarized entries into 'SPEAKER_00: text' lines."""
    entries = result.get("diarized_transcript", {}).get("entries", [])
    if not entries:  # fallback: no diarization returned
        text = result.get("transcript", "").strip()
        return f"SPEAKER_00: {text}" if text else ""

    lines = []
    for e in entries:
        text = e.get("transcript", "").strip()
        if not text:
            continue
        try:
            label = f"SPEAKER_{int(e.get('speaker_id', 0)):02d}"
        except (ValueError, TypeError):
            label = f"SPEAKER_{e.get('speaker_id')}"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Step 3: LLM post-correction (Sarvam-M)
# --------------------------------------------------------------------------
def strip_think(text: str) -> str:
    """sarvam-m is a reasoning model; remove its <think>...</think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def llm_correct(client: SarvamAI, transcript: str) -> str:
    lines = transcript.split("\n")
    total = (len(lines) + LLM_CHUNK_LINES - 1) // LLM_CHUNK_LINES
    output: list[str] = []

    for i in range(0, len(lines), LLM_CHUNK_LINES):
        chunk_lines = lines[i:i + LLM_CHUNK_LINES]
        chunk = "\n".join(chunk_lines)
        print(f"  Correcting chunk {i // LLM_CHUNK_LINES + 1}/{total}...")

        try:
            resp = client.chat.completions(
                model="sarvam-m",
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                temperature=0.05,
                max_tokens=4096,
            )
            fixed = strip_think(resp.choices[0].message.content.strip())
        except Exception as exc:
            print(f"  LLM call failed ({exc}); keeping original chunk.")
            fixed = ""

        # Safety check: if output looks truncated/summarized, keep the original
        if not fixed or len(fixed.split()) < len(chunk.split()) * LLM_MIN_RATIO:
            output.extend(chunk_lines)
        else:
            output.extend(fixed.split("\n"))

    return "\n".join(output)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Punjabi call transcription with Sarvam AI")
    parser.add_argument("audio", type=Path, help="path to audio file")
    parser.add_argument("--speakers", type=int, default=2, help="number of speakers (default 2)")
    parser.add_argument("--no-llm", action="store_true", help="skip LLM correction")
    args = parser.parse_args()

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        sys.exit("Error: SARVAM_API_KEY not set (add it to a .env file)")
    if not args.audio.exists():
        sys.exit(f"Error: file not found: {args.audio}")
    if args.audio.suffix.lower() not in AUDIO_EXTS:
        sys.exit(f"Error: unsupported format. Use one of {sorted(AUDIO_EXTS)}")

    client = SarvamAI(api_subscription_key=api_key)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clean_audio = tmp / "clean.wav"

        print("[1/3] Preprocessing audio...")
        preprocess_audio(args.audio, clean_audio)

        print("[2/3] Transcribing with Sarvam...")
        result = transcribe(client, clean_audio, tmp / "out", args.speakers)
        raw = format_transcript(result)

    if not raw:
        sys.exit("No speech detected.")

    final = raw
    if not args.no_llm:
        print("[3/3] Correcting with Sarvam LLM...")
        final = llm_correct(client, raw)
    else:
        print("[3/3] LLM correction skipped.")

    out_file = args.audio.with_suffix(".transcript.txt")
    out_file.write_text(final, encoding="utf-8")

    print("\n" + "=" * 60)
    print(final)
    print("=" * 60)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()