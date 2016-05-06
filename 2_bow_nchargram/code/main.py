__author__ = 'luoshalin'

import sys
import random
from data_process import get_utterances, cross_validate


def main(argv):

    # PARAMETERS
    # read file
    param_fpath = sys.argv[1]
    f = open(param_fpath)
    param_lines = f.readlines()
    f.close()

    # save to param dictionary
    param_dic = dict()
    for param_l in param_lines:
        param_dic[param_l.split('=')[0].strip()] = param_l.split('=')[1].strip()

    # set params
    # category & fold number
    task_cat_path = param_dic['task_cat_path']
    nontask_cat_path = param_dic['nontask_cat_path']
    fold_num = int(param_dic['fold_num'])
    classify_method = param_dic['classify_method']
    # utterances
    if param_dic['utt_type'] == 'correct':
        task_utt_path = param_dic['task_correct_utt_path']
        nontask_utt_path = param_dic['nontask_correct_utt_path']
    elif param_dic['utt_type'] == 'error':
        task_utt_path = param_dic['task_error_utt_path']
        nontask_utt_path = param_dic['nontask_error_utt_path']

    # GET UTTERANCES
    utterances = get_utterances(task_utt_path, task_cat_path, nontask_utt_path, nontask_cat_path)
    # shuffle utterances
    random.shuffle(utterances)

    # TRAIN & CROSS VALIDATION
    cross_validate(utterances, fold_num, classify_method)


if __name__ == '__main__':
    main(sys.argv)
