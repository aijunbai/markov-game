# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from abc import ABCMeta, abstractmethod
from utils import timeit
import time
import numpy as np


__author__ = 'Aijun Bai'


class Game(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, gamma, H):
        self.name = name
        self.gamma = gamma
        self.H = H
        self.t = 0
        self.players = {}
        self.state = None
        self.verbose = False
        self.animation = False

    def add_player(self, player):
        assert player.no == 0 or player.no == 1
        self.players[player.no] = player

    def configuration(self):
        return '{}({}, {})'.format(self.name, self.players[0].name, self.players[1].name)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_animation(self, animation):
        self.animation = animation

    @abstractmethod
    def numactions(self, no):
        pass

    @timeit
    def run(self):
        assert len(self.players) == 2
        assert self.state is not None

        print('configuration: {}'.format(self.configuration()))

        for self.t in range(self.H):
            if self.verbose:
                print('step: {}'.format(self.t))

            actions = np.array([
                self.players[0].act(self.state),
                self.players[1].act(self.state)], dtype=np.int)
            state_prime, rewards = self.simulate(actions)

            for j, player in self.players.items():
                if player.train:
                    player.update(
                        self.state, actions[j], actions[1 - j], rewards[j], state_prime)

            self.state = state_prime
            if self.animation:
                time.sleep(0.25)

        for player in self.players.values():
            player.done()

    @abstractmethod
    def simulate(self, actions):  # state, actions -> state, reward
        pass
