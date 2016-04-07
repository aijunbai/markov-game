# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import numpy as np

__author__ = 'Aijun Bai'

class Strategy(object):
    def __init__(self, n, pi=None):
        self.numactions = n

        if pi is not None:
            self.pi = np.array(pi)
        else:
            self.pi = np.random.dirichlet([1] * self.numactions)

    def sample(self):
        ret = np.random.multinomial(1, self.pi)
        return [k for k, v in enumerate(ret) if v > 0][0]

    def update(self, pi):
        s = sum(pi)
        if s > 1.0:
            pi = [x / s for x in pi]

        self.pi = pi

    def add_noise(self):  # this is problemetic
        pi = np.random.dirichlet([1] * self.numactions)
        self.pi += 0.01 * (pi - self.pi)

    def __str__(self):
        return str(self.pi)

    def __repr__(self):
        return self.__str__()
