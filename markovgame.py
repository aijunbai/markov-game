# coding=utf-8

from abc import ABCMeta, abstractmethod

import game

__author__ = 'Aijun Bai'


class Simulator(object, metaclass=ABCMeta):
    def __init__(self, numactions):
        self.numactions = numactions

    @abstractmethod
    def step(self, state, actions, verbose=False):
        pass

    @abstractmethod
    def initial_state(self):
        pass


class MarkovGame(game.Game, metaclass=ABCMeta):
    def __init__(self, name, simulator, gamma, H):
        super().__init__(name, gamma, H)
        self.simulator = simulator
        self.state = self.simulator.initial_state()

    def simulate(self, actions, verbose=False):
        return self.simulator.step(self.state, actions, verbose=verbose)

    @abstractmethod
    def numactions(self, no):
        pass
