# coding=utf-8


# for python 2.5

import numpy as np

__author__ = 'Aijun Bai'

class BiMatrix(object):
    def __init__(self, A=None, B=None):
        A = np.mat(A)
        B = -A if B is None else np.mat(B)
        self._bimatrix = {0: A, 1: B}

    def get_reward_for(self, i, actions):
        return self._bimatrix[i][actions[0], actions[1]]

    def numactions(self):
        return self._bimatrix[0].shape

    def matrix(self):
        return self._bimatrix
