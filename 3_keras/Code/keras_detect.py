# keras_detect.py:
# 1. process original data into features 
# 2. split into folds 
# 3. dump into pickle files

import random
import sys
from data_process import get_utterances, get_folded_data, output_data, get_vocabulary


def main(argv):
    # process parameters
    param_fpath = argv[1]
    f = open(param_fpath)
    param_lines = f.readlines()
    param_dic = dict()
    for l in param_lines:
        param_dic[l.split('=')[0].strip()] = l.split('=')[1].strip()

    task_cat_path = param_dic['task_cat_path']
    task_utt_correct_path = param_dic['task_utt_correct_path']
    task_utt_error_path = param_dic['task_utt_error_path']
    nontask_cat_path = param_dic['nontask_cat_path']
    nontask_utt_correct_path = param_dic['nontask_utt_correct_path']
    nontask_utt_error_path = param_dic['nontask_utt_error_path']
    output_folder_path = param_dic['output_folder_path']
    chat_path = param_dic['chat_path']
    dataset_type = param_dic['dataset_type']
    feature_type = param_dic['feature_type']
    fold = int(param_dic['fold_num'])  # 5-fold cv

    if dataset_type == 'error':
        task_utt_path = task_utt_error_path
        nontask_utt_path = nontask_utt_error_path
    elif dataset_type == 'correct':
        task_utt_path = task_utt_correct_path
        nontask_utt_path = nontask_utt_correct_path

    # get utterances
    utterances = get_utterances(task_cat_path, task_utt_path, nontask_cat_path, nontask_utt_path, chat_path)
    # shuffle
    random.shuffle(utterances)
    # get folded data (#fold equally splitted datasets)
    fold_data = get_folded_data(utterances, fold)
    # dumpy to pickle files
    output_data(fold_data, fold, feature_type, output_folder_path)

    # get the total voc size - for deep learning param
    voc = get_vocabulary(utterances)
    print 'total voc size: ' + str(len(voc))

    
if __name__ == '__main__':
    main(sys.argv)
