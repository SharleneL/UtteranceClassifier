__author__ = 'luoshalin'

import numpy as np


def main():
    dic_fpath = '../data/ch2v_word_list'
    vec_fpath = '../data/ch2v_vector_list'

    f = open(vec_fpath)
    lines = f.readlines()
    vec_list = []
    for line in lines:
        line_list = line.split()
        line_list = [float(x) for x in line_list]
        vec_list.append(np.array([line_list]))

    # vec_list.append([1, 4, 6])
    vec_table = np.array(vec_list)
    print vec_table[0].shape
    np.save('ch2v.npy', vec_table)

if __name__ == '__main__':
    main()
