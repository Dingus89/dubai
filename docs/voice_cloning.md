Voice Cloning Guide (Orpheus & VibeVoice) -- quickstart
    -----------------------------------------------------
    
    Overview
    --------
    We want a consistent voice format and a simple workflow for cloning voices, extracting embeddings,
    and making the pipeline choose voices automatically.
    
    Voice file naming convention (recommended)
    -----------------------------------------
    Use: {sex}_{age}_{descriptor}.{ext}
    Examples:
     - male_30_smooth.wav
     - female_50_sharp.wav
     - male_old_gravely.wav
    
    Store voices in: voices/{voice_id}/
    Example:
     voices/
       male_30_smooth/
         samples/
           sample01.wav
           sample02.wav
         metadata.json
    
    metadata.json format
    {
      "id": "male_30_smooth",
      "sex": "male",
      "age": 30,
      "desc": "smooth",
      "language": "en",
      "notes": ""
    }
    
    Steps to create a cloned voice (high level)
    -------------------------------------------
    1. Collect clean single-speaker WAVs (10-60 seconds total recommended for good results).
    2. Use the provided clone scripts to extract embeddings or fine-tune a small adapter (depends on model).
    3. Save the voice profile in voices/{voice_id}/ and add mapping to voices/voice_map.json
    
    Assign persistent voice mapping
    ------------------------------
    We included `scripts/assign_persistent_voice.py` and `scripts/clone_*` helpers.
    The function assign_persistent_voice(speaker_name, suggested) should be called in your pipeline
    to assign a voice id to new speakers. It will persist mappings in voices/voice_map.json.
    
    Orpheus cloning (Coqui placeholder)
    -----------------------------------
    - If using Coqui TTS or Orpheus, you can generate speaker embeddings (if model supports) by
      extracting mel-spectrograms and running the embedding model supplied with the TTS repo.
    - For Coqui, look for models that support speaker adaptation or fine-tuning.
    
    VibeVoice cloning
    -----------------
    - VibeVoice 1.5B is heavier; preferred approach is to extract a voice embedding using the official method and store it.
    - The pipeline can route synthesis to the VibeVoice engine for voices that match VibeVoice IDs.
    
    Files in this repo
    ------------------
     - scripts/clone_orpheus_voice.py  (starter: extracts mel, optional embed)
     - scripts/clone_vibevoice_voice.py (starter: extracts embeddings for VibeVoice)
     - voices/voice_map.json           (persistent mapping)
     - models/tts_orpheus.py           (TTS wrapper)
     - models/tts_vibevoice.py         (not included, add later)
    
    Practical tips
    --------------
     - Use 16k or 22.05k mono WAVs for consistency; Coqui TTS often expects 22k or 24k depending on model.
     - Normalize volume before cloning (e.g., `pyloudnorm` or ffmpeg -af loudnorm).
     - Label files clearly and keep 10--60s clean audio per voice for good clones.
