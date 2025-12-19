import os
import json
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel

VOICE_DIR = "voices/"
PROFILE_PATH = os.path.join(VOICE_DIR, "vibevoice_profiles.json")
MODEL_ID = "microsoft/VibeVoice-1.5B"


def extract_embeddings():
    model = AutoModel.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    profiles = {}
    for f in os.listdir(VOICE_DIR):
        if not f.endswith(".wav"):
            continue
        vid = os.path.splitext(f)[0]
        wav, sr = torchaudio.load(os.path.join(VOICE_DIR, f))
        inputs = processor(wav.squeeze(), sampling_rate=sr,
                           return_tensors="pt")
        with torch.no_grad():
            emb = model(
                **inputs).last_hidden_state.mean(1).cpu().numpy().tolist()
        profiles[vid] = emb
        print(f"✅ Extracted embedding for {vid}")
    json.dump(profiles, open(PROFILE_PATH, "w"), indent=2)
    print(f"💾 Saved embeddings to {PROFILE_PATH}")


if __name__ == "__main__":
    extract_embeddings()
