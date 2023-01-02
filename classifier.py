"""This scripts uses trained model to predict the sentiment of a review."""

import argparse
from model.logistic_regression import BinaryLogisticRegression, BLRFileManager
from model.words2numbers import Words2Numbers, W2NFileManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the sentiment of a review.")
    parser.add_argument(
        "-w",
        "--weights",
        nargs="+",
        required=True,
        help="Provide the weights file location.",
    )
    parser.add_argument(
        "-r",
        "--review",
        nargs="+",
        required=True,
        help="Provide the review message.",
    )
    parser.add_argument(
        "-m",
        "--mapper",
        nargs="+",
        required=True,
        help="Provide the mapper (words to vector) file location.",
    )
    args = parser.parse_args()

    # Load the model weights.
    weights = BLRFileManager.load_weights(args.weights[0])

    # Convert the review to a vector.
    mapper = W2NFileManager.load_feature_dictionary(args.mapper[0])
    review: str = " ".join(args.review)
    review_numeric = Words2Numbers(mapper).convert(review)

    # Predict the sentiment of the review.
    blg = BinaryLogisticRegression()
    blg.set_weights(weights)
    prediction = blg.predict(review_numeric)

    print("The sentiment of the review is", "positive" if prediction else "negative", end=".\n")
