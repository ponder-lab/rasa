"""Train the NLU EmbeddingIntentClassifier on the bundled demo data, then parse with it.

A small, self-contained driver for the training path in rasa/utils/tf_models.py. It uses the
classifier's defaults, which train eagerly, with small growing batches so the batch size, padded
sequence length and sparse feature count vary from call to call.
"""
import os
import tempfile
import time

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from rasa.nlu.training_data import load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

start = time.time()
training_data = load_data(os.path.join(ROOT, "data", "examples", "rasa", "demo-rasa.md"))
config = RasaNLUModelConfig({"language": "en", "pipeline": [
    {"name": "WhitespaceTokenizer"},
    {"name": "CountVectorsFeaturizer"},
    {"name": "EmbeddingIntentClassifier", "epochs": 20, "batch_size": [8, 32]},
]})
trainer = Trainer(config)
trainer.train(training_data)

# Parse through a persisted and reloaded model: the classifier sets its predict function only in
# load(), so parsing straight after an in-memory train fails at this version.
with tempfile.TemporaryDirectory() as model_dir:
    interpreter = Interpreter.load(trainer.persist(model_dir))
    for text in ("hello there", "i am looking for an indian spot",
                 "show me chinese restaurants in the north"):
        print("%s -> %s" % (text, interpreter.parse(text)["intent"]))
print("elapsed %.1fs" % (time.time() - start))
