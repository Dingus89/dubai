import logging
from diadub.mixing.mixer import Mixer

logging.basicConfig(level=logging.INFO)


def main():
    mixer = Mixer()
    bg = "data/samples/bg.wav"
    dialog = "data/samples/dialog.wav"
    out = "data/samples/mixed.wav"
    mixer.mix(bg, dialog, out)


if __name__ == "__main__":
    main()
