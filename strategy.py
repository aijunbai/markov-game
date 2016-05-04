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
        prob = np.sum(self.pi)
        if prob > 1.0:
            self.pi /= prob
        ret = np.random.multinomial(1, self.pi)
        return [k for k, v in enumerate(ret) if v > 0][0]

    def update(self, pi):
        self.pi = pi

    def add_noise(self):  # this is problemetic
        alpha, epsilon = 1000, 0.0001
        parameters = alpha * self.pi + np.full(self.pi.shape, epsilon)
        pi = np.random.dirichlet(parameters)
        self.pi = pi

    def __str__(self):
        return str(self.pi)

    def __repr__(self):
        return self.__str__()
