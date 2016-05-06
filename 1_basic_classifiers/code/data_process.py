__author__ = 'luoshalin'
# functions to do data preprocessing

import nltk
import random
from nltk.tokenize import TweetTokenizer
from nltk.collocations import *
from sklearn.feature_extraction import DictVectorizer


# function to get feature list X and label list y
def get_train_data(param_fpath):
    # PROCESS PARAMS
    f = open(param_fpath)
    param_lines = f.readlines()
    for l in param_lines:
        if l.split('=')[0].strip() == 'wgram_list':
            if l.split('=')[1].strip() == '':
                wgram_list = []
            else:
                wgram_str_list = l.split('=')[1].strip().split(',')
                wgram_list = [int(x) for x in wgram_str_list]
        elif l.split('=')[0].strip() == 'cgram_list':
            if l.split('=')[1].strip() == '':
                cgram_list = []
            else:
                cgram_str_list = l.split('=')[1].strip().split(',')
                cgram_list = [int(x) for x in cgram_str_list]
        elif l.split('=')[0] == 'task_fpath':
            task_fpath = l.split('=')[1].strip()
        elif l.split('=')[0] == 'nontask_fpath':
            nontask_fpath = l.split('=')[1].strip()

    # GET UTTERANCES
    utterances = []
    # read task file
    f = open(task_fpath, 'r')
    category = 'task'
    for line in f:
        line = line.lower().decode('latin-1')
        # add word & char gram
        get_utterances(utterances, line, category, wgram_list, cgram_list)
    # read nontask file
    f = open(nontask_fpath, 'r')
    category = 'nontask'
    for line in f:
        line = line.lower().decode('latin-1')
        # add word & char gram
        get_utterances(utterances, line, category, wgram_list, cgram_list)

    random.shuffle(utterances)
    # print utterances[0]

    # GET FEATURE & LABEL LISTS
    X = list(u[0] for u in utterances)  # X - ngram word & ngram character list
    y = list(u[1] for u in utterances)  # y - label list

    # convert to features
    X = [get_features(x) for x in X]    # feature list
    v = DictVectorizer()
    X = v.fit_transform(X)

    # split train & test dataset
    # train_len = int(len(features)*0.8)
    # train_set, test_set = features[:train_len], features[train_len:]
    # X_train, X_test = X[:train_len], X[train_len:]
    # y_train, y_test = y[:train_len], y[train_len:]
    return X, y


# function to convert input string into ngram word & ngram character
def get_utterances(utterances, line, category, wgram_param_list, cgram_param_list):
    tknzr = TweetTokenizer()
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    # WORD GRAMS
    wgram_list = []
    for wgram_num in wgram_param_list:
        if wgram_num == 1:  # unigram
            # unigram list
            wgram_list += tknzr.tokenize(line)
        elif wgram_num == 2:  # bigram
            # unigram list
            tokens = nltk.wordpunct_tokenize(line)
            # bigram list
            finder = BigramCollocationFinder.from_words(tokens)
            scored = finder.score_ngrams(bigram_measures.raw_freq)
            bigram_list = sorted(bigram for bigram, score in scored)
            wgram_list += [" ".join(bigram) for bigram in bigram_list]  # convert to bigram string
        elif wgram_num == 3:
            # unigram list
            tokens = nltk.wordpunct_tokenize(line)
            # trigram list
            tri_finder = TrigramCollocationFinder.from_words(tokens)
            tri_scored = tri_finder.score_ngrams(trigram_measures.raw_freq)
            trigram_list = sorted(trigram for trigram, triscore in tri_scored)
            wgram_list += [" ".join(trigram) for trigram in trigram_list]  # convert to bigram string

    # CHAR GRAMS
    cgram_list = []
    for cgram_num in cgram_param_list:
        if cgram_num == 1:    # uni-chargram
            cgram_list += [line[i:i+1] for i in range(len(line)-1)]
        elif cgram_num == 2:  # bi-chargram
            cgram_list += [line[i:i+2] for i in range(len(line)-1)]
        elif cgram_num == 3:
            cgram_list += [line[i:i+3] for i in range(len(line)-1)]

    # APPEND RESULT TO UTTERANCE LIST
    if category == 'task':
        utterances.append((wgram_list + cgram_list, 0))
    else:
        utterances.append((wgram_list + cgram_list, 1))


# function to convert list of tokens into feature
def get_features(value):
    words = set(value)
    features = {}
    for w in words:
        features[w] = 1
    return features