import sys
import csv
import numpy as np


class W2NFileManager:
    """
    This class implements functions needed to read and write files.
    """

    @staticmethod
    def load_tsv_dataset(file: str) -> np.ndarray:
        """Loads raw data and returns a tuple containing the reviews and their ratings.

        Parameters
        ----------
        file : str
            File path to the dataset tsv file.

        Returns
        -------
        np.ndarray
            An np.ndarray of shape N. N is the number of data points in the tsv file.
            Each element dataset[i] is a tuple (label, review), where the label is
            an integer (0 or 1) and the review is a string.
        """
        dataset = np.loadtxt(
            file, delimiter="\t", comments=None, encoding="utf-8", dtype="l,O"
        )
        return dataset

    @staticmethod
    def load_feature_dictionary(file: str) -> dict:
        """Creates a map of words to vectors using the file that has the word2vec
        embeddings.

        Parameters
        ----------
        file : str
            File path to the word2vec embedding file.

        Returns
        ----------
        dict
            A dictionary indexed by words, returning the corresponding word2vec
            embedding np.ndarray.
        """
        word2vec_map = dict()
        with open(file, mode="r", encoding="utf-8") as file:
            read_file = csv.reader(file, delimiter="\t")
            for row in read_file:
                word, embedding = row[0], row[1:]
                word2vec_map[word] = np.array(embedding, dtype=float)
        return word2vec_map

    @staticmethod
    def write_mapped_file(mapped_dataset: list, where_to_write: str) -> None:
        """It writes the dataset into a file.

        Parameters
        ----------
        mapped_dataset : list
            A list of tuples which are labels and mapped words.
        where_to_write : str
            A file location

        Returns
        -------
        None
        """
        with open(where_to_write, "w", encoding="utf-8") as file:
            for label, review in mapped_dataset:
                file.write(
                    f"{label:.6f}\t"
                    + "\t".join([f"{feature:.6f}" for feature in review])
                )
                file.write("\n")


class Words2Numbers:
    """This class converts the data as words to fixed-length array
    of number via word embedding mapping.
    """

    VECTOR_LEN = 300

    def __init__(self, word_mapper: dict) -> None:
        """Creates a DataPreparation instance.

        Parameters
        ----------
        word_mapper : dict
            It contains the words as key and their corresponding
            fixed-length array of numbers. Use word2vec.txt if
            you have no clue what to do.
        """
        self.word_to_num_mapper: dict = word_mapper

    def convert(self, review: str) -> np.ndarray:
        """Converts the review into a fixed-length array of numbers.

        Parameters
        ----------
        review : str
            A review as a string.

        Returns
        -------
        np.ndarray
            A fixed-length array of numbers.
        """
        return self.map(np.array([review]))

    def map(self, dataset: np.ndarray) -> list:
        """Converts the dataset with words into dataset with numbers.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset with words.

        Returns
        -------
        list
            A list of reviews with their mapped numbers.
        """
        cleaned_dataset = self.clean_non_mapped_words(dataset)
        return self.handle_mapping(cleaned_dataset)

    def handle_mapping(self, clean_dataset) -> list:
        """It sums the word embeddings and gets their average for each review.

        Parameters
        ----------
        clean_dataset : list
            A list of reviews based on real words.
            Use clean_non_mapped_words() to create the list.

        Returns
        -------
        list
            A list of reviews with their mapped numbers.
        """
        dataset_with_numbers = []
        if len(clean_dataset[0]) == 2:
            for label, review in clean_dataset:
                weights_combined = np.zeros(self.VECTOR_LEN)
                for word in review.split():
                    weights_combined += self.word_to_num_mapper[word]
                dataset_with_numbers.append((label, weights_combined / len(review.split())))
        else:
            for review in clean_dataset:
                weights_combined = np.zeros(self.VECTOR_LEN)
                for word in review.split():
                    weights_combined += self.word_to_num_mapper[word]
                dataset_with_numbers.append(weights_combined / len(review.split()))
        return dataset_with_numbers

    def clean_non_mapped_words(self, dataset: np.ndarray) -> list:
        """It cleans the non-mapped words from the dataset given.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset of words or sentences.

        Returns
        -------
        list
            A list of words or sentences which can be mapped.
        """
        cleaned_dataset = []
        if len(dataset.shape) == 2:
            for label, review in dataset:
                cleaned_review = []
                for word in review.split():
                    if word in self.word_to_num_mapper:
                        cleaned_review.append(word)
                cleaned_dataset.append((label, " ".join(cleaned_review)))
        else:
            for review in dataset:
                cleaned_review = []
                for word in review.split():
                    if word in self.word_to_num_mapper:
                        cleaned_review.append(word)
                cleaned_dataset.append(" ".join(cleaned_review))
        return cleaned_dataset


class Words2NumbersProgram:
    """
    This class implements methods used by the program itself.
    """

    def __init__(self) -> None:
        raise SystemExit("[ERROR] This class is not meant to be instantiated.")

    @staticmethod
    def main() -> None:
        """
        Main application core.
        """
        settings = Words2NumbersProgram.argument_parser()

        # Load the file contents.
        mapper = W2NFileManager.load_feature_dictionary(settings["word2vec_file"])
        train_dataset = W2NFileManager.load_tsv_dataset(settings["train"]["input_file"])
        test_dataset = W2NFileManager.load_tsv_dataset(settings["test"]["input_file"])
        validation_dataset = W2NFileManager.load_tsv_dataset(
            settings["validation"]["input_file"]
        )

        # Map the words into numbers.
        preperor = Words2Numbers(mapper)
        train_dataset_mapped = preperor.map(train_dataset)
        test_dataset_mapped = preperor.map(test_dataset)
        validation_dataset_mapped = preperor.map(validation_dataset)

        # Save the dataset with numbers.
        W2NFileManager.write_mapped_file(
            train_dataset_mapped, settings["train"]["output_file"]
        )
        W2NFileManager.write_mapped_file(
            test_dataset_mapped, settings["test"]["output_file"]
        )
        W2NFileManager.write_mapped_file(
            validation_dataset_mapped, settings["validation"]["output_file"]
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
        if len(args) != 7:
            raise SystemExit("[ERROR] Invalid number of arguments.")

        return {
            "word2vec_file": args[0],
            "train": {
                "input_file": args[1],
                "output_file": args[4],
            },
            "test": {
                "input_file": args[2],
                "output_file": args[5],
            },
            "validation": {
                "input_file": args[3],
                "output_file": args[6],
            },
        }


if __name__ == "__main__":
    Words2NumbersProgram.main()
