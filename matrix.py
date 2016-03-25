# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

__author__ = 'Aijun Bai'

class BiMatrix(object):
    def __init__(self, A, B):
        self._bimatrix = {0: A, 1: B}

    def get_reward_for(self, i, actions):
        return self._bimatrix[i][actions[0], actions[1]]

    def numactions(self):
        return self._bimatrix[0].shape
