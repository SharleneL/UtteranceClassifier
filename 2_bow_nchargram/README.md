# Introduction
This experiment uses a new algorithm for building features for utterances:
- If testing data tokens are all in training set tokens: use BOW as feature to do classification
- If there exists testing data: use character 3-gram as feature to do classification
Also modify the dataset: uses corrected utterances as well as the errorful utterances, in order to make comparisons.


**the classification accuracy is really high, reason: the dataset is small, so we can change binary classification into multiclassification; another is that the data is small, so there would be a lot of tokens out of the training vocabulary, so probably most of the utterances are classified using BOW&ncgram feature**

# Folders & files
- `data/` folder
 - Contains task & nontask datasets
 - Both `task/` and `nontask/` folder including the correct data, error data, and category data, each line of these three files are corresponding.
- `code/` folder
 - `main.py`: the file containing program's main function
 - `data_process.py`: the file containing all functions to process data
 - `parameters.txt`: the file containing all the parameters

# Run instructions
- Open `code/parameters.txt`, modify the parameters. Parameter name and value are separated by "=". Following are the parameter instructions:
    - `task_cat_path`: file path of task category file
    - `task_correct_utt_path`: file path of correct task utterances file
    - `task_error_utt_path`: file path of error task utterances file
    - `nontask_cat_path`: file path of nontask category file
    - `nontask_correct_utt_path`: file path of correct nontask utterances file
    - `nontask_error_utt_path`: file path of error nontask utterances file
    - `utt_type`: options: `[correct|error]`; specifies which dataset to classify
    - `fold_num`: an integer value; specifies the folder amount when doing cross validation
    - `classify_method`: options: `[binary|multi]`; specifies to do binary classification or multiple classification
- Under `code/` folder, run `python main.py [parameter_file_path]`