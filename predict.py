"""
CLI Prediction Tool
"""

from config import ModelConfig
from src.inference import Inference


def print_result(result):

    if result["is_unknown"]:

        print("\n🤷 UNKNOWN")

    else:

        print(
            f"\n📂 Categories : {', '.join(result['categories'])}"
        )

        print(
            "📊 Confidence : "
            + ", ".join(
                f"{c:.3f}"
                for c in result["confidences"]
            )
        )

    print("\nRaw Scores")

    print("-" * 40)

    for label, score in result["raw_scores"].items():

        print(f"{label:<15}: {score:.3f}")


def main():

    print("=" * 60)
    print("Document Classifier CLI")
    print("=" * 60)

    predictor = Inference(
        model_dir=ModelConfig.MODEL_DIR
    )

    print("\nModel loaded successfully.")
    print("Type 'exit' to quit.\n")

    while True:

        text = input("Input Text > ").strip()

        if text.lower() in [
            "exit",
            "quit",
            "q"
        ]:
            break

        if len(text) == 0:
            continue

        result = predictor.predict(text)

        print_result(result)

        print()


if __name__ == "__main__":
    main()