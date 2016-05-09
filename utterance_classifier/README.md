# UtteranceClassifier

Author: Shalin Luo

## Running Instruction
1. Open `code/main.py`, set the `utterance` variable as a list of utterances strings to be predicted
2. Load classifier model and dictionary
    - Classifier: 
        - `lib/classifier_bin/task_nontask_classifier.pkl` is the binary classifier
        - `lib/classifier_multi/task_nontask_classifier.pkl` is the multi-classifier;
    - Dictionary:
        - `lib/dic/token_index_dic.pickle` is a dictionary; key is a token string, value is a list containing token index and token occurence time in training data.
3. Call `classify()` function to get the predict result list
    - For binary classifier:
        - Output `0`: QA 
        - Output `1`: task
    - For multi-classifier:
        - Output `0`: QA
        - Output `1`: Shopping
        - Output `2`: Travel
        - Output `3`: Hotel
        - Output `4`: Food
        - Output `5`: Art 
        - Output `6`: Weather
        - Output `7`: Friends
        - Output `8`: Chat