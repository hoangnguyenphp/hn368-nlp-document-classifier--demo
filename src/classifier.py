import json
import os

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)


class TextClassifier:
    """
    Wrapper around HuggingFace Transformer.

    Responsibilities:
        - Load tokenizer
        - Load model
        - Save model
        - Restore model

    NOT responsible for:
        - Training
        - Validation
        - Prediction
    """

    def __init__(
        self,
        model_name,
        num_labels,
        label_names,
        max_length,
        device=None
    ):

        self.model_name = model_name
        self.num_labels = num_labels
        self.label_names = label_names
        self.max_length = max_length

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )

        self.model.to(self.device)

    # ------------------------------------------------------------------

    def save_model(self, output_dir):

        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir)

        self.tokenizer.save_pretrained(output_dir)

        metadata = {

            "model_name": self.model_name,

            "num_labels": self.num_labels,

            "label_names": self.label_names,

            "max_length": self.max_length
        }

        metadata_path = os.path.join(
            output_dir,
            "training_metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=4
            )

        print(f"✅ Model saved to: {output_dir}")

    # ------------------------------------------------------------------

    @classmethod
    def load_model(
        cls,
        model_dir,
        device=None
    ):
        """
        Load a trained model.

        model_dir can be either:

            - Local folder
                ./models/saved_model

            - Hugging Face Hub
                hn368/document-classifier
        """

        # --------------------------------------------------------------
        # Locate metadata
        # --------------------------------------------------------------

        if os.path.isdir(model_dir):

            metadata_path = os.path.join(
                model_dir,
                "training_metadata.json"
            )

        else:

            metadata_path = hf_hub_download(
                repo_id=model_dir,
                filename="training_metadata.json"
            )

        # --------------------------------------------------------------
        # Read metadata
        # --------------------------------------------------------------

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        # --------------------------------------------------------------
        # Create classifier instance
        # --------------------------------------------------------------

        classifier = cls(

            model_name=model_dir,

            num_labels=metadata["num_labels"],

            label_names=metadata["label_names"],

            max_length=metadata["max_length"],

            device=device
        )

        classifier.model.eval()

        return classifier