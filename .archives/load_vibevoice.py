# Load model directly
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(
    "microsoft/VibeVoice-1.5B", dtype="auto")

# Specify your desired cache directory
cache_directory = "/home/krispyai/dubAI/diadub/models/tts"
