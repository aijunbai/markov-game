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

    def __init__(self, name, gamma, max_steps):
        self.name = name
        self.gamma = gamma
        self.is_symmetric = False  # the game is symmetric for either side
        self.max_steps = max_steps
        self.t = 0
        self.players = {}
        self.state = None
        self.verbose = False
        self.animation = False

    def add_player(self, i, player):
        self.players[i] = player

    def configuration(self):
        return '{}({}, {})'.format(self.name, self.players[0].name, self.players[1].name)

    def set_verbose(self, verbose):
        self.verbose = verbose

    def set_animation(self, animation):
        self.animation = animation

    @abstractmethod
    def report(self):
        pass

    @abstractmethod
    def symmetric_state(self, state):
        pass

    @abstractmethod
    def symmetric_action(self, action):
        pass

    @abstractmethod
    def numactions(self, id_):
        pass

    @timeit
    def run(self, modes):
        assert len(self.players) == 2
        assert self.state is not None

        print('configuration: {}'.format(self.configuration()))

        for t in range(self.max_steps):
            self.t = t

            if self.verbose:
                print('step: {}'.format(t))

            actions = np.array(
                [self.players[0].act(self.state, modes[0], 0, self),
                 self.players[1].act(self.state, modes[1], 1, self)],
                dtype=np.int8)
            state_prime, rewards = self.simulate(actions)

            for j, player in self.players.items():
                if modes[j]:
                    player.update(
                        self.state,
                        actions[j],
                        actions[1 - j],
                        rewards[j],
                        state_prime,
                        j,
                        self)

            self.state = state_prime
            if self.animation:
                time.sleep(0.25)

        self.report()
        for j, player in self.players.items():
            player.done(j, self)

    @abstractmethod
    def simulate(self, actions):  # state, actions -> state, reward
        pass
