__author__ = 'luoshalin'

# This file is STEP 1:
# 1. build <word, word_vec> dictionary
# 2. load sentence, find word vectors for each word 
# 3. calculate the mean of word vectors as the sentence vector and save into filesda

import re
import numpy as np


def gen_sentvec_file(corpus_word_path, corpus_word_vector_path, task_utt_fpath, output_fpath, gen_method):
    # got the <word, vector> dict
    word_vec_dic = dict()
    with open(corpus_word_path) as wf:
        with open(corpus_word_vector_path) as vf:
            w_line = wf.readline().strip()
            v_line = vf.readline().strip()
            while w_line != '' and v_line != '':
                vec = [float(x) for x in v_line.split()]
                word_vec_dic[w_line] = vec
                w_line = wf.readline().strip()
                v_line = vf.readline().strip()

    # calculate sentence vector & save to file
    with open(output_fpath, 'w') as f_output:
        with open(task_utt_fpath) as utt_f:
            utt_line = utt_f.readline()
            while utt_line != '':
                utt_line = re.sub('[^0-9a-zA-Z\s]+', '', utt_line).rstrip().lower()  # preprocess
                # print utt_line
                word_list = utt_line.split()  # contains all words in one sentence
                vec_list = []

                for word in word_list:        # each word in the sentence
                    vec = word_vec_dic[word]
                    vec_list.append(vec)
                # sentence_vec = [elem/100 for elem in sentence_vec]

                # ------- / OPTION1: sentence vector = mean of word vectors / ------- #
                if gen_method == 'mean':
                    sentence_vec = [float(sum(col))/len(col) for col in zip(*vec_list)]  # sentence vector = average of word vectors
                # ------- / OPTION2: sentence vector = max/min/mean/var of the inner products of the word vectors / ------- #
                elif gen_method == 'inner_prd':
                    # if single word sentence
                    if len(word_list) == 1:
                        sentence_vec = [1.0] * 4
                    else:
                        # calculate normalized inner product for each pair of vectors
                        vec_M = np.array(vec_list)                                           # vector matrix for one sentence; each row is a word vector
                        prd_M = np.dot(vec_M, vec_M.T)                                       # inner product matrix for vector pairs; Mij is the inner product for <v_i, v_j>
                        len_v = np.sqrt(np.sum(np.square(vec_M), axis=1))                    # the length(squared sum) for each vector(row)
                        len_M = np.dot(np.array([len_v]).T, np.array([len_v]))               # length product matrix; Mij is the product of lengths of <v_i, v_j>
                        norm_inprd_M = np.divide(prd_M, len_M)                               # normalized inner product matrix; Mij is the normalized inner product of <v_i, v_j>
                        # save all normalized inner products into a list
                        norm_inprd_list = []
                        for i in range (1, norm_inprd_M.shape[0]):
                            norm_inprd_list.extend(norm_inprd_M.diagonal(i))
                        # convert to np array
                        norm_inprd_list = np.array(norm_inprd_list)
                        # print utt_line
                        # print norm_inprd_list
                        # compute: max, min, mean, var
                        sentence_vec = []
                        sentence_vec.append(np.max(norm_inprd_list))
                        sentence_vec.append(np.min(norm_inprd_list))
                        sentence_vec.append(np.mean(norm_inprd_list))
                        sentence_vec.append(np.var(norm_inprd_list))

                # output
                output_str = (" ".join(str(elem) for elem in sentence_vec)) + '\n'
                f_output.write(output_str)
                utt_line = utt_f.readline()