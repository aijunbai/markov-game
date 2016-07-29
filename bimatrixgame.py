# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import pprint

import numpy as np

import game
import matrix

__author__ = 'Aijun Bai'


class BiMatrixGame(game.Game):
    def __init__(self, name, gamma, max_steps):
        super().__init__(name, gamma, max_steps)
        self.bimatrix = None
        self.state = 0

    def numactions(self, a):
        return self.bimatrix.numactions()[a]

    def set_matrix(self, R=None):
        self.bimatrix = matrix.BiMatrix(R=R)
        for p in range(2):
            print('matrix[{}]:'.format(p), pprint.pformat(self.bimatrix.matrix()[p]))

    def simulate(self, actions):
        rewards = np.array([self.bimatrix.get_reward_for(i, actions) for i in range(2)])
        if rewards[0] != rewards[1]:
            self.wins[np.argmax(rewards)] += 1
            self.new_episode = True
            self.report()

        return self.state, rewards


class PenaltyShoot(BiMatrixGame):
    def __init__(self, max_steps):
        super().__init__('penaltyshoot', 0.95, max_steps)
        self.set_matrix(
            R=[[-1, 1],
               [1, -1]])


class RockPaperScissors(BiMatrixGame):
    def __init__(self, max_steps):
        super().__init__('rockpaperscissors', 0.95, max_steps)
        self.set_matrix(
            R=[[0, -1, 1],
               [1, 0, -1],
               [-1, 1, 0]])


class RockPaperScissorsSpockLizard(BiMatrixGame):
    def __init__(self, max_steps):
        super().__init__('rockpaperscissorsspocklizard', 0.95, max_steps)
        self.set_matrix(
            R=[[0, -1, 1, -1, 1],
               [1, 0, -1, 1, -1],
               [-1, 1, 0, -1, 1],
               [1, -1, 1, 0, -1],
               [-1, 1, -1, 1, 0]])


class MatchingPennies(BiMatrixGame):
    def __init__(self, max_steps):
        super().__init__('matchingpennies', 0.95, max_steps)
        self.set_matrix(
            R=[[1, -1],
               [-1, 1]])


class Inspection(BiMatrixGame):
    def __init__(self, max_steps):
        super().__init__('inspection', 0.95, max_steps)
        self.set_matrix(
            R=[[-1, 1],
               [1, -1]])


class RandomGame(BiMatrixGame):
    def __init__(self, rows, cols, max_steps):
        super().__init__('random{}x{}'.format(rows, cols), 0.95, max_steps)
        R = RandomGame.random_reward(rows, cols)
        self.set_matrix(R=R)

    @staticmethod
    def random_reward(n, m):
        high = np.full([n, m], 1.0)
        low = np.full([n, m], -1.0)
        return low + (high - low) * np.random.rand(n, m)
