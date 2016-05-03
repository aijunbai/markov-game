# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

from abc import ABCMeta, abstractmethod


import game

__author__ = 'Aijun Bai'


class Simulator(object):
    __metaclass__ = ABCMeta

    def __init__(self, g):
        self.game = g

    @abstractmethod
    def numactions(self, no):
        pass

    @abstractmethod
    def step(self, state, actions, verbose=False):
        pass

    @abstractmethod
    def initial_state(self):
        pass


class MarkovGame(game.Game):
    __metaclass__ = ABCMeta

    def __init__(self, name, simulator, gamma, H):
        super().__init__(name, gamma, H)
        self.simulator = simulator
        self.state = self.simulator.initial_state()

    def simulate(self, actions, verbose=False):
        return self.simulator.step(self.state, actions, verbose=verbose)

    @abstractmethod
    def numactions(self, no):
        pass
