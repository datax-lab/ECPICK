import numpy as np


class SeqEncoder:
    def __init__(self):
        self.categories = np.array(list('ACDEFGHIKLMNPQRSTVWYX'))

    def char_to_one_hot_encoding(self, c):
        X_int = np.zeros(len(self.categories), dtype=np.int8)
        index = np.where(self.categories == c)[0]

        if len(index) == 0: return X_int

        X_int[index] = 1
        return X_int

    def seq_to_one_hot_encoding(self, seq):
        return np.array([self.char_to_one_hot_encoding(c) for c in seq])

    def one_hot_encoding_to_seq(self, one_hot_encoding):
        X = np.array(one_hot_encoding).reshape(-1, 21)
        last_index = np.where(np.sum(X, axis=1) == 0)[0]
        last_index = 1000 if len(last_index) == 0 else last_index[0]
        return ''.join(self.categories[X.argmax(axis=1)][:last_index])
