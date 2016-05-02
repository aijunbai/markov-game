# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from numba import jit
import numpy as np

__author__ = 'Aijun Bai'

@jit(nopython=True)
def jit_normalize(pi):
    prob = np.sum(pi)
    if prob > 1.0:
        pi /= prob

class Strategy(object):
    def __init__(self, n, pi=None):
        if pi is not None:
            self.pi = np.array(pi)
        else:
            self.pi = np.random.dirichlet([1] * n)

    def sample(self):
        jit_normalize(self.pi)
        ret = np.random.multinomial(1, self.pi)
        return [k for k, v in enumerate(ret) if v > 0][0]

    def update(self, pi):
        self.pi = np.array(pi)

    def add_noise(self):  # this is problemetic
        alpha, epsilon = 1000, 0.0001
        parameters = alpha * self.pi + np.full(self.pi.shape, epsilon)
        pi = np.random.dirichlet(parameters)
        self.pi = pi

    def __str__(self):
        return str(self.pi)

    def __repr__(self):
        return self.__str__()
