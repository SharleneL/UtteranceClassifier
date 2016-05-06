__author__ = 'luoshalin'

from helper import get_dic, analyze


def main():
    # FILE PATHS
    dic_fpath = '../data/birkbeck.txt'
    data_dir = '../data/data'

    # ANALYZE THE DATASET & SAVE CORRECTED DATA
    error_correct_dic, correct_set = get_dic(dic_fpath)
    analyze(data_dir, error_correct_dic, correct_set)


if __name__ == '__main__':
    main()
