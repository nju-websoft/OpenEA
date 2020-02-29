import time

import numpy as np

import read

word2vec_file = ''


def load_vectors(fname):
    t = time.time()
    fin = read.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    print("num of word vectors:", n)
    print("dim:", d)
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([list(map(float, tokens[1:]))], dtype=np.float64)
        # print(data[tokens[0]])
    assert len(data) == n
    print("load word vectors cost: {:.3f} s".format(time.time() - t))
    return data
