__author__ = 'luoshalin'

from nltk.tokenize import word_tokenize
import cPickle
import os


# ==========/ LOAD DATA INTO UTTERANCES[] /========== #
def get_utterances(task_cat_path, task_utt_path, nontask_cat_path, nontask_utt_path, chat_path):
    utterances = []
    # save the task utterances - (utterance_line, category)
    with open(task_cat_path) as f_cat, open(task_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip()
            utt_line = "^" + utt_line.strip().replace(" ", "#") + "$"
            utterances.append((utt_line, cat_line))
    print 'After appending task utt: ' + str(len(utterances))

    # save the non-task utterances - (utterance_line, category)
    with open(nontask_cat_path) as f_cat, open(nontask_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip()
            utt_line = utt_line.strip()
            utterances.append((utt_line, cat_line))
    print 'After appending nontask utt: ' + str(len(utterances))

    # save the chat data - (utterance_line, category)
    with open(chat_path) as f:
        for line in f:
            line = line.strip()
            utterances.append((line, 'Chat'))
    print 'After appending chat utt: ' + str(len(utterances))

    return utterances


# ==========/ GET FEATURES FOR CROSS-VALIDATION TRAIN-TEST SETS /========== #
def output_data(fold_data, fold, feature_type, output_folder_path):
    # delete current files under output folder
    for the_file in os.listdir(output_folder_path):
        file_path = os.path.join(output_folder_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    # generate new files
    indexed_voc = dict()

    for i in range(0, fold):
        # get train & test sets for current iteration
        test_utterances = fold_data[i]
        train_utterances = []
        for j in range(0, fold):
            if j != i:
                train_utterances += fold_data[j]
        # get the vocabulary of train & test sets
        train_voc = get_vocabulary(train_utterances)
        test_voc = get_vocabulary(test_utterances)
        # check whether testing tokens are all included in training tokens => decide whether to use ngram
        ngram = False   # to mark whether to use ngram-char or not
        for test_token in test_voc:
            if test_token not in train_voc:  # if any of the test token did not appear in the train set, use ngram char
                ngram = True
                break

        # get train & test features
        train_feature = get_features(train_utterances, feature_type)
        test_feature = get_features(test_utterances, feature_type)
        print 'train feature size: ' + str(len(train_feature))
        print 'test feature size: ' + str(len(test_feature))
        # split - train
        train_X = [k[0] for k in train_feature]
        train_X_indexed = get_indexed_X(train_X, indexed_voc)
        train_y = [k[1] for k in train_feature]
        # split - test
        test_X = [k[0] for k in test_feature]
        test_X_indexed = get_indexed_X(test_X, indexed_voc)
        test_y = [k[1] for k in test_feature]

        # save indexed feature to pickle file
        cPickle.dump(train_X_indexed, open(output_folder_path + "train_X_"+str(i)+".cPickle","w"))
        cPickle.dump(test_X_indexed, open(output_folder_path + "test_X_"+str(i)+".cPickle","w"))
        cPickle.dump(train_y, open(output_folder_path + "train_y_"+str(i)+".cPickle","w"))
        cPickle.dump(test_y, open(output_folder_path + "test_y_"+str(i)+".cPickle","w"))


# ==========/ SPLIT CROSS-VALIDATION TRAIN-TEST SETS /========== #
def get_folded_data(utterances, fold):
    fold_data = []
    fold_len = len(utterances) / fold
    for i in range(0, 5):
        fold_data.append(utterances[i*fold_len: (i+1)*fold_len])
    return fold_data


# ==========/ HELPER FUNCTIONS /========== #
def get_vocabulary(utterances):
    token_list = []
    for utt in utterances:
        utt_content = utt[0]
        token_list += word_tokenize(utt_content)
    token_set = set(token_list)
    return token_set


def get_features(utterances, feature_type):
    features = []
    for utt in utterances:
        utt_content = utt[0]  # text content of the utterance
        utt_category = utt[1]

        # ========== Only BOW - start ========== #
        bow_list = word_tokenize(utt_content)
        feature_list = bow_list
        # ========== Only BOW - end ========== #

        # ========== OPT*1 : Add 3GRAM - start ========== #
        if feature_type == 'bow_trigram':
            cgram_list = [utt_content[i:i+3] for i in range(len(utt_content)-1)] # 3-gram list
            feature_list += cgram_list
        # ========== OPT*1 - end ========== #

        # ========== OPT*2: USE BOW & 3GRAM CONDITIONALLY(whether train voc cover all test voc) - start ========== #
        # if ngram:  # use bow & ngram as feature
        #     # add 3-gram char list
        #     cgram_list = [utt_content[i:i+3] for i in range(len(utt_content)-1)] # 3-gram list
        #     feature_list += cgram_list
        # ========== OPT*2 - end ========== #

        if utt_category == 'QA':            # non-task
            features.append((feature_list, 0))
        elif utt_category == 'Shopping':    # task
            features.append((feature_list, 1))
        elif utt_category == 'Travel':      # task
            features.append((feature_list, 2))
        elif utt_category == 'Hotel':       # task
            features.append((feature_list, 3))
        elif utt_category == 'Food':        # task
            features.append((feature_list, 4))
        elif utt_category == 'Art':         # task
            features.append((feature_list, 5))
        elif utt_category == 'Weather':     # task
            features.append((feature_list, 6))
        elif utt_category == 'Friends':     # task
            features.append((feature_list, 7))
        elif utt_category == 'Chat':        # chat
            features.append((feature_list, 8))
        else:
            print utt_category,"ERROR"
    return features


def get_indexed_X(features, index_voc):
    indexed_features = []
    for feature_list in features:
        indexed_feature_list = []
        for token in feature_list:
            if token in index_voc:
                indexed_feature_list.append(index_voc[token])
            else:
                index_voc[token] = len(index_voc)
                indexed_feature_list.append(index_voc[token])
        indexed_features.append(indexed_feature_list)
    return indexed_features