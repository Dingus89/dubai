from diadub.models.registry import ModelRegistry

reg = ModelRegistry("models.json", device="cuda")
m = reg.get("asr_whisper_emotion")
# This returns an instance but does NOT load weights until:
m.load()  # will attempt GPU load and fallback as needed
out = m.infer("data/samples/test.wav")
m.unload()
