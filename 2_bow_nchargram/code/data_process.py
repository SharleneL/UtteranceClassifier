__author__ = 'luoshalin'

import nltk
from nltk.tokenize import TweetTokenizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.svm import SVC


# FUNCTION TO LOAD DATA INTO UTTERANCES[]
def get_utterances(task_utt_path, task_cat_path, nontask_utt_path, nontask_cat_path):
    utterances = []
    # save the task utterances - (utterance_line, category)
    with open(task_cat_path) as f_cat, open(task_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip()
            utt_line = utt_line.strip()
            utterances.append((utt_line, cat_line))

    # save the non-task utterances - (utterance_line, category)
    with open(nontask_cat_path) as f_cat, open(nontask_utt_path) as f_utt:
        for cat_line, utt_line in zip(f_cat, f_utt):
            cat_line = cat_line.strip()
            utt_line = utt_line.strip()
            utterances.append((utt_line, cat_line))
    return utterances


# FUNCTION TO CREATE CROSS-VALIDATION TRAIN-TEST SETS & DO CROSS-VALIDATION
def cross_validate(utterances, fold_num, classify_method):
    # create cv train & test datasets
    fold_len = len(utterances) / fold_num
    fold_data = []
    for i in range(0, 5):
        fold_data.append(utterances[i*fold_len: (i+1)*fold_len])

    # do cv
    for i in range(0, fold_num):
        # get train & test sets for current iteration
        test_utterances = fold_data[i]
        train_utterances = []
        for j in range(0, fold_num):
            if j != i:
                train_utterances += fold_data[j]
        # get the vocabulary of train & test sets
        train_voc = get_vocabulary(train_utterances)
        test_voc = get_vocabulary(test_utterances)
        # check whether testing tokens are all included in training tokens => decide whether to use ngram
        ngram = False                           # to mark whether to use ngram-char or not
        for test_token in test_voc:
            if test_token not in train_voc:     # if any of the test token did not appear in the train set, use ngram char
                ngram = True
                break

        # get train & test features
        train_features = get_features(train_utterances, ngram, classify_method)  # a list of (token_list, class_num) tuples
        test_features = get_features(test_utterances, ngram, classify_method)

        train_X_tklists = [f[0] for f in train_features]        # a list of token_list
        test_X_tklists = [f[0] for f in test_features]

        tk_idx_dic = get_tk_idx_dic(train_X_tklists + test_X_tklists)
        train_X = get_trainM(train_X_tklists, tk_idx_dic)       # convert list of token_list to sparse matrix
        test_X = get_trainM(test_X_tklists, tk_idx_dic)

        train_y = np.asarray([f[1] for f in train_features])    # a list of class_num(0/1)
        test_y = np.asarray([f[1] for f in test_features])

        # # LR
        # lr = LogisticRegression(C=1.0)
        # lr.fit(train_X, train_y)
        # lr_y_pred = lr.predict(test_X)
        # lr_acry = metrics.accuracy_score(test_y, lr_y_pred)
        # print '[LR Classification] Cross Validation Fold#' + str(i+1) + ': ' + str(lr_acry)

        # SVM
        for C in [1, 10, 50, 100]:
            for gamma in [0.01, 0.1, 0.5, 1.0]:
                svm = SVC(C=C, gamma=gamma)
                svm.fit(train_X, train_y)
                svm_y_pred = svm.predict(test_X)
                svm_acry = metrics.accuracy_score(test_y, svm_y_pred)
                print 'C='+str(C) + '\tgamma='+str(gamma)
                print '[SVM Classification] Cross Validation Fold#' + str(i+1) + ': ' + str(svm_acry)


# FUNCTION TO GET VOC FOR UTTERANCES
def get_tk_idx_dic(tk_lists):
    dic = dict()
    # token_list = []
    idx = 0
    for tk_l in tk_lists:
        for token in tk_l:
            if token not in dic:
                dic[token] = [idx, 1]
                idx += 1
            else:
                dic[token][1] += 1
    return dic


def get_vocabulary(utterances):
    token_list = []
    for utt in utterances:
        utt_content = utt[0]
        token_list += nltk.wordpunct_tokenize(utt_content)
    token_set = set(token_list)
    return token_set


# FUNCTION TO GET FEATURES FOR ONE UTTERANCE
def get_features(utterances, ngram, classify_method):
    features = []
    tknzr = TweetTokenizer()
    for utt in utterances:
        utt_content = utt[0]  # text content of the utterance
        utt_category = utt[1]

        if ngram:  # use bow & ngram as feature
            # bow list
            bow_list = tknzr.tokenize(utt_content)
            # cgram list
            uni_cgram_list = [utt_content[i:i+1] for i in range(len(utt_content)-1)]
            bi_cgram_list = [utt_content[i:i+2] for i in range(len(utt_content)-1)]
            tri_cgram_list = [utt_content[i:i+3] for i in range(len(utt_content)-1)]
            feature_list = bow_list         # add bow tokens
            feature_list += uni_cgram_list  # add unigram character lists
            feature_list += bi_cgram_list   # add bigram character lists
            feature_list += tri_cgram_list  # add trigram character lists
        else:  # only use bow as feature
            feature_list = tknzr.tokenize(utt_content)

        if classify_method == 'binary':
            if utt_category == 'QA':  # non-task
                features.append((feature_list, 0))
            else:  # task
                features.append((feature_list, 1))
        elif classify_method == 'multi':
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


def get_trainM(tklists, tk_idx_dic):
    row = []
    col = []
    data = []

    for i in range(len(tklists)):
        tk_l = tklists[i]
        for t in tk_l:
            row.append(i)
            col.append(tk_idx_dic[t][0])
            data.append(tk_idx_dic[t][1])
    trainM = csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(len(tklists), len(tk_idx_dic)))
    return trainM