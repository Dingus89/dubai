import logging
from diadub.models.registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("debug")


def main():
    registry = ModelRegistry()
    translator = registry.get("translation")

    test_sentences = [
        "Bonjour, comment allez-vous ?",
        "Je suis très heureux de vous rencontrer.",
        "Ceci est un test de traduction automatique.",
    ]
    results = translator.batch_translate(test_sentences)
    for src, out in zip(test_sentences, results):
        log.info(f"{src}  -->  {out}")


if __name__ == "__main__":
    main()
