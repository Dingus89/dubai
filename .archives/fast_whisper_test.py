from diadub.models.stt.whisper_emotion import WhisperEmotionSTT
from diadub.models.stt.wav2vec_emotion import Wav2VecEmotionSTT

emo = Wav2VecEmotionSTT()
asr = WhisperEmotionSTT(emotion_model=emo)
asr.load()

out = asr.infer("samples/test.wav", analyze_emotions=True)
print(out["segments"][:3])
