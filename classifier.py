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
        required=False,
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
        required=False,
        help="Provide the mapper (words to vector) file location.",
    )
    args = parser.parse_args()

    # Load the model weights.
    if args.weights is None:
        weights_file: str = "best_weights.tsv"
        print("No weights file provided. Using the default weights file:", weights_file)
    else:
        weights_file = args.weights[0]

    weights = BLRFileManager.load_weights(weights_file)

    # Convert the review to a vector.
    if args.mapper is None:
        mapper_file: str = "dataset/word2vec.txt"
        print("No mapper file provided. Using the default mapper file:", mapper_file)
    else:
        mapper_file = args.mapper[0]

    mapper = W2NFileManager.load_feature_dictionary(mapper_file)
    review: str = " ".join(args.review)
    review_numeric = Words2Numbers(mapper).convert(review)

    # Predict the sentiment of the review.
    blg = BinaryLogisticRegression()
    blg.set_weights(weights)
    prediction = blg.predict(review_numeric)

    print("The sentiment of the review is", "positive" if prediction else "negative", end=".\n")
