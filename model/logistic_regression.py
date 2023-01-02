import sys
import numpy as np


class BLRFileManager:
    """
    This class implements functions needed to read and write files.
    """

    @staticmethod
    def save_errors(errors: dict, file_to_write: str) -> None:
        """Writes given errors to the file.

        Parameters
        ----------
        errors : dict
            A dict instance wit keys "train", "test" and "validation"
            and float values for errors.
        file_to_write : str
            A file location to write errors.
        """
        if not isinstance(errors, dict):
            raise TypeError("Given errors must be a dict instance.")

        try:
            train_error = errors.get("train")
            test_error = errors.get("test")
            validation_error = errors.get("validation")
        except KeyError as exc:
            raise KeyError(
                "Given errors dictionary must have train, test and validation keys."
            ) from exc

        string_to_write = f"error(train): {train_error:.6f}\n"
        string_to_write += f"error(test): {test_error:.6f}\n"
        string_to_write += f"error(validation): {validation_error:.6f}"

        with open(file_to_write, "w", encoding="utf-8") as output_file:
            output_file.write(string_to_write)

    @staticmethod
    def save_predictions(predictions: np.ndarray, file_to_write: str) -> None:
        """It saves the Numpy ndarray to a TXT file.

        Parameters
        ----------
        predictions : np.ndarray
            Prediction np.ndarray
        file_to_write : str
            The file location to write predictions.
        """
        np.savetxt(
            fname=file_to_write,
            X=predictions,
            fmt="%d",
            newline="\n",
        )
    
    @staticmethod
    def save_weights(weights: np.ndarray, file_to_write: str) -> None:
        """It saves the weights numpy.ndarray to a file.

        Parameters
        ----------
        weights : np.ndarray
            The weights of the model.
        file_to_write : str
            The file location tot write weights.
        """
        np.savetxt(
            fname=file_to_write,
            X=weights,
            fmt="%f",
            newline="\n",
        )


class BLRDatasetReader:
    """
    This class is a reader for the number-based dataset files.
    """

    def __init__(self, dataset_location: str = None, dataset_list: list = None) -> None:
        if dataset_location is not None:
            self.file = np.loadtxt(dataset_location, delimiter="\t")
        else:
            self.file = np.array(dataset_list)

    @property
    def all(self) -> np.ndarray:
        """Returns the all data.

        Returns
        -------
        np.ndarray
            The all data.
        """
        return self.file

    @property
    def features(self) -> np.ndarray:
        """Returns the features.

        Returns
        -------
        np.ndarray
            The features.
        """
        return self.file[:, 1:]

    @property
    def labels(self) -> np.ndarray:
        """Returns the labels.

        Returns
        -------
        np.ndarray
            The labels.
        """
        return self.file[:, 0]


class BinaryLogisticRegression:
    """
    A binary logistic model trainer.
    """

    def __init__(self) -> None:
        # Internal attributes.
        self._weights: np.ndarray = None
        self._is_trained: bool = False
        self._num_epoch: int = None
        self._learning_rate: float = None

    def train(
        self,
        input_matrix: np.ndarray,
        label_vector: np.ndarray,
        num_epoch: int,
        learning_rate: int,
        add_intercept: bool = True,
    ) -> tuple:
        """Trains the model using the given input vectors and labels,
        it performs the stochastic gradient descent algorithm.

        Parameters
        ----------
        input_matrix : np.ndarray
            Input matrix of features.
        label_vector : np.ndarray
            Input vector of real labels for features in input_matrix.
        num_epoch : int
            The number of times the training will be done for dataset.
        learning_rate : int
            Learning rate for the model optimizer (GD).
        add_intercept : bool, optional
            Should program add additional bias term, by default True

        Returns
        -------
        tuple
            A tuple of (errors, predictions_list)
        """
        # Copy the input matrix to avoid modifying the original one.
        c_input_matrix = input_matrix.copy()

        # Add a bias term to the input matrix.
        if add_intercept:
            c_input_matrix = np.insert(c_input_matrix, 0, 1, axis=1)

        # Initialize the weight vector with zeros.
        if self._weights is None:
            self._weights = np.zeros(c_input_matrix.shape[1])

        # Do the training for epoches.
        for _ in range(num_epoch):
            predictions = self.predict(
                c_input_matrix, weights=self._weights, add_bias=False
            )
            prediction_difference_from_labels = predictions - label_vector
            gradient = np.dot(prediction_difference_from_labels.T, c_input_matrix)
            self._weights = self._weights - learning_rate * gradient

        self._is_trained = True
        return (self.compute_error(predictions, label_vector), predictions)

    def get_weights(self) -> np.ndarray:
        """Returns weights of the model.

        Returns
        -------
        np.ndarray
            Returns weights of the model.
        """
        if self._weights is not None and self._is_trained:
            return self._weights

    def predict(
        self,
        input_matrix: np.ndarray,
        weights: np.ndarray = None,
        add_bias: bool = True,
    ) -> np.ndarray:
        """Predicts the labels for the given input vectors.

        Parameters
        ----------
        input_matrix : np.ndarray
            Input matrix of features.
        weights : np.ndarray, optional
            The weighs for the Binary Logistic Regression model,
            if not given, trained weights would be used. (by default None)
        add_bias : bool, optional
            Should program add additional bias term, by default True

        Returns
        -------
        np.ndarray
            Predictions for given input matrix.

        Raises
        ------
        SystemExit
            Rasied when model is not trained and weights are not given.
        """
        # Check if the model is trained.
        if not self._is_trained and weights is None:
            raise SystemExit("[ERROR] Model is not trained yet.")

        # Use the trained weights if no weights are provided.
        if weights is None:
            weights = self._weights

        # Copy the input matrix to avoid modifying the original one.
        c_input_matrix = input_matrix.copy()

        # Add a bias term to the input matrix.
        if add_bias:
            c_input_matrix = np.insert(c_input_matrix, 0, 1, axis=1)

        resulting_vector = np.dot(c_input_matrix, weights)
        sigmoid_of_results = self.sigmoid(resulting_vector)
        return np.where(sigmoid_of_results >= 0.5, 1, 0)

    def calculate_cross_entropy_error(
        self, predicted_labels: np.ndarray, real_labels: np.ndarray
    ) -> float:
        # @TODO: Implement this!
        """Calculates the cross entropy error for the given input vectors and labels.

        Parameters
        ----------
        predicted_labels : np.ndarray
            Predicted labels with trained model.
        real_labels : np.ndarray
            Real labels from dataset.

        Returns
        -------
        float
            Cross entropy error of the model.
        """
        return -0.1

    @staticmethod
    def compute_error(predicted_labels: np.ndarray, real_labels: np.ndarray) -> float:
        """Computes the error between the predicted and the actual labels.

        Parameters
        ----------
        predicted_labels : np.ndarray
            Predicted labels with trained model.
        real_labels : np.ndarray
            Real labels from dataset.

        Returns
        -------
        float
            Mean absolute error of model.
        """
        return np.mean(np.abs(predicted_labels - real_labels))

    @staticmethod
    def sigmoid(input_vec: np.ndarray) -> np.ndarray:
        """Implementation of the sigmoid function.

        Parameters
        ----------
        input_vec : np.ndarray
            Input element of the sigmoid function.

        Returns
        -------
        np.ndarray
            sigmoid(x) where x is input_vec
        """
        euler = np.exp(input_vec)
        return euler / (1 + euler)


class BLGTrainerProgram:
    """
    This class implements methods used by the program itself.
    """

    def __init__(self) -> None:
        raise SystemExit("[ERROR] This class is not meant to be instantiated.")

    @staticmethod
    def main():
        """
        Main application core.
        """
        settings = BLGTrainerProgram.argument_parser()

        # Read the datasets prepared with feature.py script.
        train_dataset = BLRDatasetReader(settings["train"]["input_file"])
        test_dataset = BLRDatasetReader(settings["test"]["input_file"])
        validation_dataset = BLRDatasetReader(settings["validation"]["input_file"])

        # Train the model.
        blg = BinaryLogisticRegression()
        train_error, train_predictions = blg.train(
            train_dataset.features,
            train_dataset.labels,
            num_epoch=int(settings["model_options"]["num_epoch"]),
            learning_rate=float(settings["model_options"]["learning_rate"]),
        )

        # Predict the labels for the test set.
        test_predictions = blg.predict(test_dataset.features)
        test_error = blg.compute_error(test_predictions, test_dataset.labels)

        # Predict the labels for the validation set.
        validation_predictions = blg.predict(validation_dataset.features)
        validation_error = blg.compute_error(
            validation_predictions, validation_dataset.labels
        )

        # Write the metrics to the output file.
        errors = {
            "train": train_error,
            "test": test_error,
            "validation": validation_error,
        }
        BLRFileManager.save_errors(errors, settings["model_options"]["metrics_output"])
        BLRFileManager.save_weights(blg.get_weights(), settings["model_options"]["weights_output"])

        # Write the predictions to the output file.
        BLRFileManager.save_predictions(
            train_predictions, settings["train"]["output_file"]
        )
        BLRFileManager.save_predictions(
            test_predictions, settings["test"]["output_file"]
        )
        BLRFileManager.save_predictions(
            validation_predictions, settings["validation"]["output_file"]
        )

    @staticmethod
    def argument_parser() -> dict:
        """It parses the arguments given to the program.

        Returns
        -------
        dict
            Argument names as keys and their values.

        Raises
        ------
        SystemExit
            Raises error if invalid number of arguments given.
        """
        args = sys.argv[1:]
        if len(args) != 10:
            raise SystemExit("[ERROR] Invalid number of arguments.")

        return {
            "model_options": {
                "metrics_output": args[6],
                "num_epoch": args[7],
                "learning_rate": args[8],
                "weights_output": args[9]
            },
            "train": {
                "input_file": args[0],
                "output_file": args[3],
            },
            "test": {
                "input_file": args[1],
                "output_file": args[4],
            },
            "validation": {
                "input_file": args[2],
                "output_file": args[5],
            },
        }


if __name__ == "__main__":
    BLGTrainerProgram.main()
