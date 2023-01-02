from words2numbers import Words2Numbers, W2NFileManager
from logistic_regression import BinaryLogisticRegression, BLRDatasetReader

datasets = {
    "train": "../dataset/train_data.tsv",
    "test": "../dataset/test_data.tsv",
    "validation": "../dataset/validation_data.tsv",
    "word2vec": "../dataset/word2vec.txt",
}

model_opts = {
    "num_epochs": 500,
    "learning_rate": 0.001
}

if __name__ == "__main__":
    # Load the file contents.
    mapper = W2NFileManager.load_feature_dictionary(datasets["word2vec"])
    train_dataset = W2NFileManager.load_tsv_dataset(datasets["train"])
    test_dataset = W2NFileManager.load_tsv_dataset(datasets["test"])
    val_dataset = W2NFileManager.load_tsv_dataset(datasets["validation"])

    # Map the words into numbers.
    preperor = Words2Numbers(mapper)
    train_dataset_mapped = preperor.map(train_dataset)
    test_dataset_mapped = preperor.map(test_dataset)
    validation_dataset_mapped = preperor.map(val_dataset)

    # Read the datasets prepared with feature.py script.
    train_dataset = BLRDatasetReader(dataset_list=train_dataset_mapped)
    test_dataset = BLRDatasetReader(dataset_list=test_dataset_mapped)
    validation_dataset = BLRDatasetReader(dataset_list=validation_dataset_mapped)

    # Train the model.
    blg = BinaryLogisticRegression()
    train_error, train_predictions = blg.train(
        train_dataset.features,
        train_dataset.labels,
        num_epoch=model_opts["num_epochs"],
        learning_rate=model_opts["learning_rate"],
    )

    # Predict the labels for the test set.
    test_predictions = blg.predict(test_dataset.features)
    test_error = blg.compute_error(test_predictions, test_dataset.labels)

    # Predict the labels for the validation set.
    validation_predictions = blg.predict(validation_dataset.features)
    validation_error = blg.compute_error(
        validation_predictions, validation_dataset.labels
    )

    # Print the metrics.
    errors = {
        "train": train_error,
        "test": test_error,
        "validation": validation_error,
    }
    print(f"Errors: {errors}", errors)

    weights = blg.get_weights()
