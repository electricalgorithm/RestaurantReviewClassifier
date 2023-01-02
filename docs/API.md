# Application User Interface

This page servers as a manual for the model files and their functionality.

### Words2Numbers Application
This is the application version of the Words2Numbers module. It is designed to be used as a standalone application.

##### Usage
```bash
~$ python words2numbers.py <map_file_txt> <train_dataset> <test_dataset> <validation_dataset> <train_output> <test_output> <validation_output>
```
* `<map_file_txt>` is the path to the mapping text file.
* `<train_dataset>` is the path to the training dataset.
* `<test_dataset>` is the path to the test dataset.
* `<validation_dataset>` is the path to the validation dataset.
* `<train_output>` is the path to the output numbered file for the training dataset.
* `<test_output>` is the path to the output numbered file for the test dataset.
* `<validation_output>` is the path to the output numbered file for the validation dataset.

### Words2Numbers
This module designed to have statistically informative inputs from given human-readable texts. You may think of it as a "text-to-number" module. It is designed to be used in a pipeline with other modules.

##### Usage
```python
from words2numbers import Words2Numbers

# Read the mapping dictionary from a file.
mapper_dict = W2NFileManager.load_feature_dictionary("mapper.txt")

# Initialize the module with mapping dictionary.
w2n = Words2Numbers(mapper_dict)

# Convert a text to a number.
sentences_numbered = w2n.map("He said that one hundred and twenty three is the destiny.")
```

### W2NFileManager
This module is designed to read and write mapping text files, reading datasets, and writing results.
1. `load_feature_dictionary` reads a mapping text file and returns a dictionary.
2. `load_tsv_dataset` reads a dataset in TSV format and returns a NumPy ndArray.
3. `write_mapped_file` writes a list of number-ed sentences to into a file.