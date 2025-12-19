from faster_whisper import WhisperModel

model_size = "large-v3"  # Or "tiny", "base", "small", "medium"
# Specify your desired cache directory
cache_directory = "/home/krispyai/dubAI/diadub/models/"

# Load the model, and it will be downloaded to the specified cache_directory
model = WhisperModel(model_size, device="cuda",
                     compute_type="float16", download_root=cache_directory)
