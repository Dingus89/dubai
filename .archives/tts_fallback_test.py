t = reg.get("tts_vibevoice")
try:
    t.load()
except Exception as e:
    print("Load failed, try CPU:", e)
    t.load(prefer_8bit=False)  # forces CPU path
wav = t.synth_text("Hello world", prosody_params={"rate": 1.0})
t.unload()
