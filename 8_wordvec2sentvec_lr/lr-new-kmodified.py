# Kyusong modified code
# This file is STEP 2:
# 1. load the sentence vectors for [task & nontask & chat] datasets, and the category for each sentence
# 2. construct feature vectors to fit in with the sk-learn LR tranining method
# 3. run sk-learn LR

import random,sys
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score


def main():
    # SET FILE PATHS
    # each line in category file corresponding to each line in vector file
    # category paths
    nontask_cat_path = '../data/utterances/nontask/nontask-category.csv'
    task_cat_path = '../data/utterances/task/task-category.csv'
    # ------ / PARAMS TO BE CHANGED - START / ------ #
    # vector paths - the files generated by gen_sentvec.py
    output_folder_path = '../output/sentvec_innerprd_4features/'
    nontask_correct_vector_fpath = output_folder_path + 'nontask_correct_sentence_vector_list'
    nontask_error_vector_fpath = output_folder_path + 'nontask_error_sentence_vector_list'
    task_correct_vector_fpath = output_folder_path + 'task_correct_sentence_vector_list'
    task_error_vector_fpath = output_folder_path + 'task_error_sentence_vector_list'
    chat_vector_fpath = output_folder_path + 'chat_sentence_vector_list'
    # ------ / PARAMS TO BE CHANGED - END / ------ #

    # LOAD DATA
    vectors = []
    # save the task vectors - (vec_line, category)
    with open(task_cat_path) as f_cat, open(task_correct_vector_fpath) as f_vec:
        for cat_line, vec_line in zip(f_cat, f_vec):
            cat_line = cat_line.strip().decode('latin-1')
            vec_line = vec_line.strip().decode('latin-1')
            get_vectors(vectors, vec_line, cat_line)
    # save the nontask vectors - (vec_line, category)
    with open(nontask_cat_path) as f_cat, open(nontask_correct_vector_fpath) as f_vec:
        for cat_line, vec_line in zip(f_cat, f_vec):
            cat_line = cat_line.strip().decode('latin-1')
            vec_line = vec_line.strip().decode('latin-1')
            get_vectors(vectors, vec_line, cat_line)
    # save the chat vectors - (vec_line, category)
    with open(chat_vector_fpath) as f:
        for line in f:
            line = line.strip().decode('latin-1')
            get_vectors(vectors, line, 'Chat')

    random.shuffle(vectors)
    X = list(v[0] for v in vectors)  # list of vectors
    y = list(v[1] for v in vectors)  # list of categories

    # convert to features
    #X = [get_features(x) for x in X]  # get features for each vector

    #print X
    #sys.stdin.readline()

    #v = DictVectorizer()
    #X = v.fit_transform(X)
    print len(X),len(y)

#     ==========/ Logistic Regression /========== 
    lr = LogisticRegression(C=1e30)
    #lr.fit(X,y)
    scores = cross_val_score(lr, X, y, cv=10, scoring='accuracy')
    print scores
    print scores.mean()
    
# HELPER FUNCTIONS
# convert to features
def get_features(vec):
    features = {}
    i = 0
    for num in vec:
        features[i] = num
        i += 1
    return features
    
def get_vectors(vectors, vec_line, category):
    vec_list = [float(x) for x in vec_line.split()]
    # RESULT
    if category == 'QA':            # non-task
        vectors.append((vec_list, 0))
    elif category == 'Shopping':    # task
        vectors.append((vec_list, 1))
    elif category == 'Travel':      # task
        vectors.append((vec_list, 2))
    elif category == 'Hotel':       # task
        vectors.append((vec_list, 3))
    elif category == 'Food':        # task
        vectors.append((vec_list, 4))
    elif category == 'Art':         # task
        vectors.append((vec_list, 5))
    elif category == 'Weather':     # task
        vectors.append((vec_list, 6))
    elif category == 'Friends':     # task
        vectors.append((vec_list, 7))
    elif category == 'Chat':        # chat
        vectors.append((vec_list, 8))
    else:
        print category,"ERROR"

if __name__ == '__main__':
    main()
    