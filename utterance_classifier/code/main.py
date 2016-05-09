__author__ = 'luoshalin'

import sys
from sklearn.externals import joblib
import pickle
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix
import numpy as np


def main():
    # --- / LOAD FILES / --- #
    # load classifier
    classifier_path = '../lib/classifier_bin/task_nontask_classifier.pkl'
    # classifier_path = '../lib/classifier_multi/task_nontask_classifier.pkl'
    classifier = joblib.load(classifier_path)
    # load dictionary
    with open('../lib/dic/token_index_dic.pickle', 'rb') as f:      # <token, [index, cunt]> dic
        tk_idx_dic = pickle.load(f)

    # --- / INPUT DATA: a list of utterances / --- #
    utterances = ['i want inbian feod',
                  'whether is good today',
                  'how much is the erreirplame ticket',
                  'What is the meaning of overhunting']

    # --- / DO CLASSIFICATION / --- #
    pred_list = classify(classifier, tk_idx_dic, utterances)

    print pred_list


# ---/ HELPER FUNCTIONS / --- #
# classification function
def classify(classifier, tk_idx_dic, utterances):
    features = get_features(utterances)  # a list of token_list
    utt_X = get_trainM(features, tk_idx_dic)
    return classifier.predict(utt_X)


# INPUT: a list of utterances string
# OUTPUT: a list of feature lists(BOW & 1, 2, 3-gram character) from
def get_features(utterances):
    features = []
    tknzr = TweetTokenizer()
    for utt_content in utterances:
        # generate bow list
        bow_list = tknzr.tokenize(utt_content)
        # genarate char-ngram list
        uni_cgram_list = [utt_content[i:i+1] for i in range(len(utt_content)-1)]
        bi_cgram_list = [utt_content[i:i+2] for i in range(len(utt_content)-1)]
        tri_cgram_list = [utt_content[i:i+3] for i in range(len(utt_content)-1)]
        feature_list = bow_list         # add bow tokens
        feature_list += uni_cgram_list  # add unigram character lists
        feature_list += bi_cgram_list   # add bigram character lists
        feature_list += tri_cgram_list  # add trigram character lists

        features.append(feature_list)

    return features


# INPUT: a list of feature lists; <token, [index, cunt]> dic
# OUTPUT: sparse matrix fitting with classifier
def get_trainM(tklists, tk_idx_dic):
    row = []
    col = []
    data = []

    for i in range(len(tklists)):
        tk_l = tklists[i]
        for t in tk_l:
            if t not in tk_idx_dic:
                continue
            row.append(i)
            col.append(tk_idx_dic[t][0])
            data.append(tk_idx_dic[t][1])

    trainM = csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(len(tklists), len(tk_idx_dic)))
    return trainM


if __name__ == '__main__':
    main()
