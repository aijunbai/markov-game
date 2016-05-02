# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from abc import ABCMeta, abstractmethod
import numpy as np
import utils

__author__ = 'Aijun Bai'


class Game(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, gamma, H):
        self.name = name
        self.gamma = gamma
        self.H = int(H)
        self.players = utils.makehash()
        self.step = 0
        self.state = None

    def add_player(self, player):
        assert player.no == 0 or player.no == 1
        self.players[player.no] = player

    def clear_players(self):
        self.players = utils.makehash()

    @abstractmethod
    def numactions(self, no):
        pass

    def run(self, verbose=False):
        assert len(self.players) == 2
        assert self.state is not None

        for self.step in range(self.H):
            if verbose: print('step: {}'.format(self.step))

            actions = np.array([
                self.players[0].act(self.state),
                self.players[1].act(self.state)], dtype=np.int)
            next_state, rewards = self.simulate(actions, verbose=verbose)

            for j, player in self.players.items():
                if player.train:
                    player.update(
                        self.state, actions[j], actions[1 - j], rewards[j], next_state)

            self.state = next_state

        for player in self.players.values():
            player.done()

    @abstractmethod
    def simulate(self, actions, verbose=False):  # state, actions -> state, reward
        pass
