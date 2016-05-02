# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
from builtins import *

import strategy

__author__ = 'Aijun Bai'


class Particle(object):
    def __init__(self, agent, policy=None, K=None):
        self.agent = agent
        self.strategy = strategy.Strategy(self.agent.numactions, policy)

        if K is not None:
            self.K = np.array(K)
        else:
            self.K = np.random.rand(self.agent.opp_numactions)

    def val(self):
        return min(self.K[o] for o in range(self.agent.opp_numactions))

    def clone(self):
        cloned = Particle(self.agent, policy=self.strategy.pi, K=self.K)
        cloned.add_noise()
        return cloned

    def add_noise(self):
        self.strategy.add_noise()

    def __str__(self):
        return str(self.strategy)

    def __repr__(self):
        return self.__str__()
