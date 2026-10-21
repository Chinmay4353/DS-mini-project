"""Command-line inference using the same trained pipeline as the dashboard."""

from src.data import CATEGORICAL_FEATURES, FEATURE_METADATA, NUMERICAL_FEATURES
from src.model import load_pipeline, predict_single_student


def prompt_student_profile():
    """Collect and return a single student profile from the terminal."""
    profile = {}
    print("Enter student details:")

    for feature in NUMERICAL_FEATURES:
        label = FEATURE_METADATA[feature]["label"]
        profile[feature] = float(input(f"{label}: "))

    for feature in CATEGORICAL_FEATURES:
        label = FEATURE_METADATA[feature]["label"]
        options = ", ".join(FEATURE_METADATA[feature]["options"])
        profile[feature] = input(f"{label} ({options}): ").strip()

    return profile


def main():
    profile = prompt_student_profile()
    prediction = predict_single_student(load_pipeline(), profile)
    print(f"\nPrediction: {prediction['label']}")
    print(f"Estimated placement probability: {prediction['probability_placed']:.1%}")
    print("This estimate is decision support, not a guarantee of an individual outcome.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, EOFError) as error:
        raise SystemExit(f"Could not make a prediction: {error}") from error