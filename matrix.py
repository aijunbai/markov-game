# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import numpy as np

__author__ = 'Aijun Bai'

class BiMatrix(object):
    def __init__(self, R=None):
        self._bimatrix = {0: np.array(R), 1: -np.array(R)}

    def get_reward_for(self, i, actions):
        return self._bimatrix[i][actions[0], actions[1]]

    def numactions(self):
        return self._bimatrix[0].shape

    def matrix(self):
        return self._bimatrix
