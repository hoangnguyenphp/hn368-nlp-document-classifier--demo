from dataclasses import dataclass

@dataclass
class ModelConfig:

    MODEL_NAME = "xlm-roberta-base"

    MAX_LENGTH = 128

    NUM_LABELS = 3

    LABELS = [
        "Economy",
        "Politics",
        "Sports"
    ]

    MODEL_DIR = "./models/saved_model"

    MODEL_REPO = "hn368/hn_first_nlp_document-classifier"


@dataclass
class TrainConfig:

    BATCH_SIZE = 4

    EPOCHS = 50

    LEARNING_RATE = 2e-5

    WARMUP_STEPS = 10

    TEST_SIZE = 0.2


@dataclass
class PredictConfig:

    THRESHOLD = 0.5

    UNKNOWN_THRESHOLD = 0.3