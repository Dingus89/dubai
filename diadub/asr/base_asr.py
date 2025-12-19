import abc
from typing import List, Dict, Any


class BaseASR(abc.ABC):
    """Abstract base class for ASR backends."""

    @abc.abstractmethod
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Transcribe an audio file and return a list of segments with:
        {
          "text": str,
          "start": float,
          "end": float,
          "confidence": float
        }
        """
        pass
