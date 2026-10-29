# Multilingual Call Transcription

A Python command-line prototype that transcribes call recordings with speaker labels. It uses Sarvam AI for Hindi and Punjabi, and ElevenLabs for English. Audio is preprocessed with FFmpeg before transcription.

## Features

- Supports `.wav`, `.mp3`, `.m4a`, `.flac`, and `.ogg` audio files
- Adds speaker labels to transcript turns
- Preprocesses audio to mono, 16 kHz, with noise filtering and loudness normalization
- Provides separate Sarvam and ElevenLabs scripts, plus an automatic routing script

## Project files

- `transcribe_auto.py` — attempts to detect the language and route the recording to the appropriate provider
- `Sarvam.py` — transcribes Hindi or Punjabi with Sarvam and can apply Punjabi text correction
- `Elevenlabs.py` — transcribes English with ElevenLabs
- `call.mp3` — sample recording, if you have permission to share it
- `call.transcript.txt` — sample transcript

## Requirements

- Python 3.10 or later
- FFmpeg installed and available on your PATH
- Sarvam AI API key
- ElevenLabs API key for English transcription

## Setup

Clone or download this repository, then run:

```bash
python -m venv .venv