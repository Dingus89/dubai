from speechbrain.inference.interfaces import foreign_class

# Specify your custom directory using the 'savedir' parameter
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    savedir="~/dubai/diadub/models/stt"  # Set your custom path here
)
