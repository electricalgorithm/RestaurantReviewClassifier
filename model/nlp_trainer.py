import argparse
import numpy as np
from words2numbers import Words2Numbers, W2NFileManager
from logistic_regression import BinaryLogisticRegression, BLRDatasetReader, BLRFileManager

class TrainingResult:
    """Represents the result of the training process."""

    def __init__(
        self,
        model_weights: np.ndarray,
        epoch_num: int,
        learning_rate: float,
        validation_error: dict,
    ):
        self.epoch_num: int = epoch_num
        self.learning_rate: float = learning_rate
        self.error: float = validation_error
        self.weights: np.ndarray = model_weights

    def __str__(self) -> str:
        """String representation of the object."""
        return f"Training Result for epoch={self.epoch_num}, learning_rate={self.learning_rate}\n\
            validation error={self.error: .6f}"

    def __repr__(self) -> str:
        """String representation of the object."""
        return self.__str__()

    def __eq__(self, other):
        """Equality operator."""
        return (
            self.learning_rate == other.learning_rate
            and self.epoch_num == other.epoch_num
        )

    def __hash__(self) -> int:
        """Hash function."""
        return hash((self.learning_rate, self.epoch_num))

    def __lt__(self, other):
        """Less than operator."""
        return self.error < other.error

    def __le__(self, other):
        """Less than or equal to operator."""
        return self.error <= other.error

    def __gt__(self, other):
        """Greater than operator."""
        return self.error > other.error

    def __ge__(self, other):
        """Greater than or equal operator."""
        return self.error >= other.error


class NLPTrainerProgram:
    """Main program to train the model."""

    @staticmethod
    def main():
        """Main core of the program."""
        # Parse the arguments.
        args = NLPTrainerProgram.argument_parser()
        # Load the file contents.
        mapper = W2NFileManager.load_feature_dictionary(args.mapper[0])
        train_dataset = W2NFileManager.load_tsv_dataset(args.train[0])
        test_dataset = W2NFileManager.load_tsv_dataset(args.test[0])
        val_dataset = W2NFileManager.load_tsv_dataset(args.validation[0])

        # Map the words into numbers.
        preperor = Words2Numbers(mapper)
        train_dataset_mapped = preperor.map(train_dataset)
        test_dataset_mapped = preperor.map(test_dataset)
        validation_dataset_mapped = preperor.map(val_dataset)

        # Read the datasets prepared with feature.py script.
        train_dataset = BLRDatasetReader(dataset_list=train_dataset_mapped)
        test_dataset = BLRDatasetReader(dataset_list=test_dataset_mapped)
        validation_dataset = BLRDatasetReader(dataset_list=validation_dataset_mapped)

        # Hyperparameters to test.
        num_epochs = [10, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000]
        learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

        # Train the model for different hyperparameters.
        results = []
        for num_epoch in num_epochs:
            for learning_rate in learning_rates:
                # Train the model.
                blg = BinaryLogisticRegression()
                blg.train(
                    train_dataset.features,
                    train_dataset.labels,
                    num_epoch=num_epoch,
                    learning_rate=learning_rate,
                )

                # Predict the labels for the validation set.
                validation_predictions = blg.predict(validation_dataset.features)
                validation_error = blg.compute_mse(
                    validation_predictions, validation_dataset.labels
                )

                result = TrainingResult(
                    blg.get_weights(),
                    num_epoch,
                    learning_rate,
                    validation_error,
                )
                # Save the results.
                results.append(result)

        # Find the best hyperparameters.
        best_result = min(results)
        print(
            f"\nBest Hyperparameter Result:\n\
                - Learning Rate: {best_result.learning_rate}\n\
                - Epochs: {best_result.epoch_num}\n\
                - Validation Error: {best_result.error: 0.6f}"
        )

        # Test the model with the best hyperparameters.
        blg = BinaryLogisticRegression()
        blg.set_weights(best_result.weights)
        test_predictions = blg.predict(test_dataset.features)
        test_error = blg.compute_mse(test_predictions, test_dataset.labels)
        print(f"\nTest Error: {test_error: 0.6f}")

        # Save the model weights.
        BLRFileManager.save_weights(best_result.weights, args.output[0])

    @staticmethod
    def argument_parser():
        """Argument parser for the program."""
        desc = "This script finds you the best hyperparameters \
               and model weights for the NLP binary logistic regression model."
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument(
            "-m",
            "--mapper",
            nargs="+",
            required=True,
            help="Provide the mapper (words to vector) file location.",
        )
        parser.add_argument(
            "-t",
            "--train",
            nargs="+",
            required=True,
            help="Provide the train dataset file location as TSV.",
        )
        parser.add_argument(
            "-s",
            "--test",
            nargs="+",
            required=True,
            help="Provide the test dataset file location as TSV.",
        )
        parser.add_argument(
            "-v",
            "--validation",
            nargs="+",
            required=True,
            help="Provide the validation dataset file location as TSV.",
        )
        parser.add_argument(
            "-o",
            "--output",
            nargs="+",
            required=True,
            help="Provide the file location to save the model weights.",
        )
        return parser.parse_args()


if __name__ == "__main__":
    NLPTrainerProgram.main()
