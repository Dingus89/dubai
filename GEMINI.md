# GEMINI.md

## Project Overview

This project, `DiaDub`, is a sophisticated, open-source AI dubbing system implemented in Python. It provides a fully automated, modular pipeline for dubbing video content. The system performs a sequence of operations including:

1.  **Audio Extraction:** Separates the audio from the video.
2.  **Vocal Separation:** Separates the dialogue from the background audio using `demucs`.
3.  **Speech Recognition (ASR):** Transcribes the dialogue using `whisperx`.
4.  **Speaker Diarization:** Identifies the different speakers in the audio using `pyannote`.
5.  **Translation:** Translates the transcribed text to the target language.
6.  **Emotion Analysis:** Analyzes the audio to get the emotional context.
7.  **Script Generation:** Creates a detailed script with information about the speakers, timing, volume, and emotion.
8.  **Voice Cloning:** Uses the separated vocal tracks to clone the voices of the original speakers.
9.  **TTS Synthesis:** Synthesizes the translated text in the cloned voices, using the generated script for timing and prosody.
10. **Forced Alignment:** Generates word-level alignments for the new dialogue using the Montreal Forced Aligner.
11. **Viseme Generation:** Creates viseme data for lip-syncing.
12. **Audio/Video Merging:** Combines the new dialogue with the background audio and the video.
13. **Lip-sync:** Applies the generated viseme data to the video to sync the mouth movements with the new dialogue.

The project is designed to be highly modular and configurable, allowing users to swap out different AI models for various stages of the pipeline. Key technologies employed include:

*   **ASR:** `whisperx`
*   **Diarization:** `pyannote`
*   **Vocal Separation:** `demucs`
*   **Translation:** Hugging Face transformers
*   **TTS:** Pluggable TTS engines like `Orpheus-TTS` and `VibeVoice`.
*   **Forced Alignment:** `Montreal Forced Aligner`
*   **Audio Processing:** `ffmpeg`, `pydub`, `librosa`.
*   **ML/AI Frameworks:** `PyTorch`, `transformers`.

The entire process is managed by a central pipeline (`diadub/pipeline.py`) that supports checkpointing and resumption of individual stages, making it robust and user-friendly.

## Building and Running

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd DiaDub
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install system dependencies:**
    ```bash
    sudo apt-get install ffmpeg
    ```

### Running the Pipeline

The main entry point for the dubbing process is `run_pipeline.py`.

```bash
python run_pipeline.py --video <path-to-video> --out <output-directory>
```

**Common flags:**

*   `--video`: Path to the input video file.
*   `--out`: Directory to save the output files.
*   `--resume`: Resume the pipeline from the last completed stage.
*   `--language`: Target language for translation (e.g., 'es', 'fr').
*   `--groq-use`: Use Groq API for script enhancement.

### Testing

To verify the installation and run a test, you can use the following command:

```bash
python run_pipeline.py --video input/korean_trailer.mp4 --out tests/out/
```

## Key Modules

*   **`diadub/pipeline.py`:** The central orchestrator of the entire dubbing process. It defines the sequence of operations, manages state through a checkpointing system, and integrates all the different modules.
*   **`scripts/run_pipeline.py`:** The main entry point for executing the pipeline. It handles command-line argument parsing and initiates the `Pipeline` class.
*   **`models.json`:** The primary configuration file that specifies which models and backends to use for the different stages of the pipeline.
*   **`diadub/models/tts_engine_manager.py`:** This class is responsible for loading and managing different TTS engines (like Orpheus and VibeVoice) and is central to the voice cloning process.
*   **`diadub/script/script_generator.py`:** This module takes ASR output and other analysis (like emotion) to generate a polished, structured script for the TTS engines.
*   **`diadub/lipsync/forced_align.py`:** This file contains the logic for performing forced alignment using the Montreal Forced Aligner.
*   **`diadub/lipsync/viseme_mapper.py`:** This module converts the word and phoneme timings from the forced aligner into visemes, the visual representation of speech sounds.

## Development Conventions

*   **Modular Architecture:** The project is structured in a highly modular way, with different functionalities isolated into their own directories (e.g., `asr`, `tts`, `translation`). When adding new features, follow this modular approach.
*   **Configuration:** AI models are configured through `models.json`.
*   **Model Loading:** The project is moving away from a centralized `ModelRegistry`. Instead, components are encouraged to load their own models. The `TTSEngineManager` is a good example of this new approach.
*   **Pipeline Stages:** The main pipeline in `diadub/pipeline.py` is divided into distinct stages. When modifying the pipeline, ensure that the changes are compatible with the existing stages and that checkpointing is handled correctly.
*   **Logging:** The project uses the `loguru` library for logging. Use the logger to provide informative messages for debugging and monitoring.
*   **Code Style:** The code follows the PEP 8 style guide for Python. Use a linter to ensure your code adheres to the style guide.
*   **Error Handling:** The pipeline includes robust error handling and fallback mechanisms. When adding new functionality, consider potential failure points and implement appropriate error handling.
*   **Dependencies:** All Python dependencies are listed in `requirements.txt`. When adding a new dependency, add it to this file.