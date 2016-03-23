# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import numpy as np

__author__ = 'Aijun Bai'

class Strategy(object):
    def __init__(self, n, pi=None):
        self.numactions = n

        if pi:
            self._pi = np.array(pi)
        else:
            self._pi = np.random.dirichlet([1] * self.numactions)

    def sample(self):
        ret = np.random.multinomial(1, self._pi)
        return [k for k, v in enumerate(ret) if v > 0][0]

    def pi(self):
        return self._pi

    def update(self, pi):
        self._pi = pi

    def __str__(self):
        return str(self._pi)

    def __repr__(self):
        return self.__str__()