# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

__author__ = 'Aijun Bai'

class BiMatrix(object):
    def __init__(self, A, B):
        self._bimatrix = {0: A, 1: B}

    def get_reward_for(self, i, a1, a2):
        return self._bimatrix[i][a1, a2]
