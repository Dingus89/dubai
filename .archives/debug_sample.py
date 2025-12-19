import logging
from diadub.models.registry import ModelRegistry
from diadub.ffmpeg_utils import extract_audio

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("debug")


def main():
    video_path = "data/samples/test.mp4"
    audio_path = extract_audio(video_path)
    registry = ModelRegistry()
    asr = registry.get("asr")
    result = asr.transcribe(audio_path)

    for seg in result[:5]:
        log.info(f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}")


if __name__ == "__main__":
    main()
