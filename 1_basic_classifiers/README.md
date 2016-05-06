**Shalin Luo * 2016**

# Introduction
This set of experiment is to:
 1. Test different machine learning model's classification accuracy on task/nontask dataset (binary classification)
 2. Test n-gram word and n-gram character feature's influence on classification accuracy

# About folders & files:
1. `data`
    - Contains task and nontask utterance files.
    - Currently, `nontask_sample.txt` is the sampled nontask utterances, in order to get an even amount of data points for each class.
2. `code`
    - `main.py`: main function for the program
    - `data_process.py`: contains functions to convert a string utterance into features and labels
    - `models.py`: contains all the machine learning models
    - `parameter.py`: contains the parameters to run the program; parameter name and value are separated by "="

# Running instructions:
    1. Open `parameters.txt` file and modify the parameters. Parameter name and value are separated by "=". Following are the instructions for parameters:
        - `wgram_list`: the list of gram number for word grams, separated by ",". For example, if we want to use both unigram word and bigram word as features, we should set "wgram_list=1,2"
        - `cgram_list`: the list of gram number for character grams, separated by ",". For example, if we want to use both unigram character and bigram character as features, we should set "wgram_list=1,2"
        - `task_fpath`: the file path for the file containing task oriented utterances
        - `nontask_fpath`: the file path for the file containing nontask oriented utterances
    2. Under `code/` folder, run `python main.py [parameter_file_path]`