# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *
import numpy as np
__author__ = 'Aijun Bai'

class Strategy(object):
    def __init__(self, n, pi=None):
        if pi is not None:
            self.pi = np.array(pi)
        else:
            self.pi = np.random.dirichlet([1] * n)

    def sample(self):
        return np.random.choice(self.pi.size, size=1, p=self.pi)[0]

    def update(self, pi):
        assert isinstance(pi, np.ndarray)
        self.pi = pi
        self.pi /= np.sum(self.pi)

    def add_noise(self):  # this is problemetic
        alpha, epsilon = 1000, 0.0001
        parameters = alpha * self.pi + np.full(self.pi.shape, epsilon)
        pi = np.random.dirichlet(parameters)
        self.pi = pi

    def __str__(self):
        return str(self.pi)

    def __repr__(self):
        return self.__str__()
