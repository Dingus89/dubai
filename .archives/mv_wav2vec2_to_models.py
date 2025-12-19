import torchaudio
from speechbrain.pretrained.interfaces import EncoderWav2vecClassifier

# Define the local path where your model files (e.g., hyperparams.yaml,
# pytorch_model.bin, label_encoder.txt, etc.) are stored.
local_path = "./local_model_path"

# Load the model from the local directory
# The `source` argument is the path to your local directory.
# You can also use `run_opts={"device":"cuda"}` if you want to use a GPU.
classifier = EncoderWav2vecClassifier.from_hparams(
    source=local_path, local_files_only=True)

print("Model loaded locally.")
