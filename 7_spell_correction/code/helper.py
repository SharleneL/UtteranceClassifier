__author__ = 'luoshalin'

import os


def get_dic(fpath):
    dic = dict()            # <error, correct> dictionary
    correct_set = set()     # all correct words

    with open(fpath) as f:
        line = f.readline()
        while line != '':
            if line[0] == '$':  # correct word
                correct_word = line.strip()[1:].replace('_', ' ')
                correct_set.add(correct_word)
            else:   # error word
                error_word = line.strip().replace('_', ' ')
                dic[error_word] = correct_word
            # read next line
            line = f.readline()

    return dic, correct_set


def analyze(data_dir, dic, correct_set):
    total_amt = 0
    error_amt = 0
    correct_amt = 0

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            fpath = os.path.join(subdir, file)
            output_fpath = '../output/output_' + file
            f = open(fpath)
            lines = f.readlines()
            with open(output_fpath, 'a') as output_f:
                for l in lines:
                    tk_list = l.strip().split()
                    for tk in tk_list:
                        total_amt += 1
                        if tk in dic and tk not in correct_set:  # error token
                            l.replace(tk, dic[tk])  # correct the sentence
                            print 'ERROR:\t' + tk + '\t' + 'CORRECT:\t' + dic[tk]
                            error_amt += 1
                        else:
                            correct_amt += 1
                    # write corrected line into new file
                    output_f.write(l)

    print 'Total token number: ' + str(total_amt)
    print 'Error token number: ' + str(error_amt)
    print 'Correct token number: ' + str(correct_amt)