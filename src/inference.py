"""
Inference logic for Text Classification
"""

import numpy as np
import torch

from config import PredictConfig
from src.classifier import TextClassifier


class Inference:

    def __init__(
        self,
        model_dir,
        device=None
    ):

        self.classifier = TextClassifier.load_model(
            model_dir=model_dir,
            device=device
        )

        self.classifier.model.eval()

    # =====================================================
    # SINGLE PREDICTION
    # =====================================================

    def predict(
        self,
        text,
        threshold=PredictConfig.THRESHOLD,
        unknown_threshold=PredictConfig.UNKNOWN_THRESHOLD
    ):

        inputs = self.classifier.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.classifier.max_length,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.classifier.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.classifier.model(**inputs)

            logits = outputs.logits

            probabilities = (
                torch.sigmoid(logits)
                .squeeze()
                .cpu()
                .numpy()
            )

        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])

        max_confidence = np.max(probabilities)

        # =====================================================
        # UNKNOWN
        # =====================================================

        if max_confidence < unknown_threshold:

            return {

                "categories": ["Unknown"],

                "confidences": [1.0],

                "is_unknown": True,

                "raw_scores": {

                    label: float(score)

                    for label, score in zip(
                        self.classifier.label_names,
                        probabilities
                    )
                }
            }

        # =====================================================
        # NORMAL PREDICTION
        # =====================================================

        categories = []

        confidences = []

        for label, score in zip(
            self.classifier.label_names,
            probabilities
        ):

            if score >= threshold:

                categories.append(label)

                confidences.append(float(score))

        if len(categories) == 0:

            idx = np.argmax(probabilities)

            categories = [
                self.classifier.label_names[idx]
            ]

            confidences = [
                float(probabilities[idx])
            ]

        return {

            "categories": categories,

            "confidences": confidences,

            "is_unknown": False,

            "raw_scores": {

                label: float(score)

                for label, score in zip(
                    self.classifier.label_names,
                    probabilities
                )
            }
        }

    # =====================================================
    # BATCH PREDICTION
    # =====================================================

    def predict_batch(

        self,

        texts,

        threshold=PredictConfig.THRESHOLD,

        unknown_threshold=PredictConfig.UNKNOWN_THRESHOLD

    ):

        results = []

        for text in texts:

            results.append(

                self.predict(

                    text,

                    threshold,

                    unknown_threshold

                )

            )

        return results