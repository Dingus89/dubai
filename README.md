DiaDub is a fully modular open-source AI dubbing system written in Python.

It automatically performs speech recognition, translation, emotion-aware script generation, timed TTS synthesis, alignment, mixing, and muxing back into video -- all with clean checkpointing and resumable stages.
* * *

## ⚙️

## Core Features

| 

Category

 | 

Description

 | 
| ---- | ----  |
| 

🎧 Audio Extraction

 | 

Uses ffmpeg to extract WAV audio from input video.

 | 
| 

🧠 ASR (Speech Recognition)

 | 

Uses Faster-Whisper for high-speed transcription.

 | 
| 

🗣️ Emotion Recognition

 | 

Optional Wav2Vec2 emotion model adds emotional context.

 | 
| 

🌍 Translation

 | 

Hugging Face transformer pipeline for multilingual translation.

 | 
| 

📝 Script Generation

 | 

Creates a structured JSON "movie script" from SRT or ASR, with volume, emotion, and timing.

 | 
| 

🧩 TTS

 | 

Pluggable text-to-speech (e.g. Orpheus-TTS, Microsoft VibeVoice 1.5B).

 | 
| 

🎚 Prosody Mapping

 | 

Analyzes and adjusts pitch, rate, and loudness automatically.

 | 
| 

🎞 Mixing & Muxing

 | 

Recombines dialogue with background and merges back into video.

 | 
| 

🔄 Checkpointing

 | 

Resume any stage safely, with artifact validation.

 | 
| 

🪄 Open-Source Models

 | 

No paid APIs required -- Groq API optional for advanced script generation.

 | 

* * *

## 🏗️

## Project Layout
    
    
    diadub/
    │
    ├── pipeline.py                # Main orchestrator (core pipeline)
    │
    ├── audio_analysis/
    │   └── audio_features.py      # Loudness, pitch, LUFS analysis
    │
    ├── script/
    │   ├── script_generator.py    # Builds emotional script JSON
    │   └── script_parser.py       # Converts script to TTS-ready items
    │
    ├── translation/
    │   └── translation_manager.py # Translation logic via Hugging Face
    │
    ├── sync/
    │   └── sync_manager.py        # Timing validation tools
    │
    ├── models/
    │   ├── registry.py            # Model registry for pluggable backends
    │   ├── tts_model.py           # TTS backend wrapper
    │   ├── asr_model.py           # Faster-Whisper backend
    │   └── audio_analyzing_model.py # Emotion analysis (Wav2Vec2)
    │
    ├── utils/
    │   ├── logging_config.py
    │   └── ffmpeg_utils.py
    │
    ├── storage/
    │   └── checkpoint.py          # Stage & artifact persistence
    │
    └── dev_context.json           # Dev summary for modules

* * *

## 🚀

## Installation

1. Clone the repository
    
    
    git clone https://github.com/yourname/diadub.git
    cd diadub

1.   

2. Set up a virtual environment
    
    
    python3 -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate

2.   

3. Install core dependencies
    
    
    pip install torch torchaudio transformers datasets librosa soundfile pyloudnorm tqdm pydub ffmpeg-python numpy

3.   

4. Install system dependencies
    
    
    sudo apt install ffmpeg rubberband

4.   

5. (Optional) Install GPU-optimized Faster-Whisper
    
    
    pip install faster-whisper

5.   

6. (Optional) Install Groq client
    
    
    pip install groq

6.   

7. Environment variables (optional)
    
    
    export GROQ_API_KEY="your_groq_api_key"

  

* * *

## 🧩

## Usage Example

  

Basic Run:
    
    
    python -m diadub.pipeline --video input.mp4 --out output/

Resume from checkpoint:
    
    
    python -m diadub.pipeline --video input.mp4 --out output/ --resume

Enable Groq API script enhancement:
    
    
    python -m diadub.pipeline --video input.mp4 --groq-use --out output/

* * *

## 🧠

## How It Works

| 

Stage

 | 

Description

 | 
| ---- | ----  |
| 

extract_audio

 | 

Extracts WAV track from the video.

 | 
| 

asr

 | 

Transcribes using Faster-Whisper.

 | 
| 

diarization

 | 

(Optional) Separates speakers.

 | 
| 

translation

 | 

Translates transcript to target language.

 | 
| 

generate_script

 | 

Builds emotion-tagged, loudness-aware script JSON.

 | 
| 

tts

 | 

Synthesizes voice lines with perfect timing and emotional prosody.

 | 
| 

assemble_dialogue

 | 

Overlays voice tracks at correct timestamps.

 | 
| 

mix

 | 

Blends dialogue and background music.

 | 
| 

mux

 | 

Merges the final dubbed audio into video.

 | 

* * *

## 🧾

## Configuration

  

All models and devices are controlled via models.json, e.g.:
    
    
    {
      "asr": "faster-whisper-large-v3",
      "emotion_model": "emotion-recognition-wav2vec2-IEMOCAP",
      "tts": "Orpheus-TTS",
      "translation": "facebook/m2m100_418M"
    }

You can freely swap out models -- just update models.json.
* * *

## 🪄

## Output

  

When completed, you'll find in your output/ folder:
    
    
    input.wav                - extracted audio
    input_translated.srt     - translated subtitles
    input_script.json        - full emotional script
    input_dialogue.wav       - dialogue-only track
    input_dialogue_mixed.wav - mixed track
    input_dub.mp4            - final dubbed video
* * *

## 🧰

## Logging and Debugging

  

Logs are written to data/cache/pipeline.log and per-stage to the checkpoint file.

You can adjust verbosity by editing:
    
    
    from diadub.utils.logging_config import setup_logging
    setup_logging(level="DEBUG")

* * *

# ✅

# Static Code Integrity Check

  

All core files (pipeline.py, new modules, utils, models) were statically checked:

| 

Check

 | 

Result

 | 
| ---- | ----  |
| 

✅ Syntax Valid

 | 

No syntax errors in any module.

 | 
| 

✅ Imports

 | 

All local imports resolved successfully.

 | 
| 

✅ FFmpeg & Rubber Band

 | 

Verified external commands used safely.

 | 
| 

✅ Missing Functions

 | 

None -- all references defined or guarded.

 | 
| 

✅ GPU Overflow Handling

 | 

Memory errors handled by catching CUDA OOM and switching to CPU fallback.

 | 
| 

✅ Cross-module consistency

 | 

Registry, prosody, and analysis modules interconnect cleanly.

 | 

All modules are load-safe and will execute under Python ≥3.9 with PyTorch installed.

* * *

# 🧩

# Setup & Test Checklist

  

### 🧱 Environment

- Python ≥ 3.9
- PyTorch installed (with CUDA if GPU available)
- ffmpeg, rubberband installed via package manager
- Virtual environment activated

  

### 🔧 Install Required Packages
    
    
    pip install -r requirements.txt

(You can create this file by exporting your pip freeze or using the install list above.)

  

### 🔍 Verify Model Loading

  

Run:
    
    
    python -m diadub.models.registry

If models load successfully, it should list:

ASR: faster-whisper-large-v3, Emotion: wav2vec2-IEMOCAP, TTS: Orpheus-TTS, etc.

  

### 🎞️ Test the Pipeline
    
    
    python -m diadub.pipeline --video tests/demo.mp4 --out tests/out/

Check for these outputs:

- demo_script.json generated
- demo_dub.mp4 plays with new audio

  

### 💾 Check Resume Logic

  

Re-run the same command.

The pipeline should skip all completed stages and finish quickly.

  

### 🧠 Optional Groq API Test

  

If you export GROQ_API_KEY, re-run with --groq-use

and verify _groq_hook.json is created in the output folder.
# dubai
