# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import numpy as np
import pprint

from builtins import *

import game
import matrix

__author__ = 'Aijun Bai'


class BiMatrixGame(game.Game):
    def __init__(self, name, gamma, H):
        super().__init__(name, gamma, H)
        self.bimatrix = None
        self.state = 0

    def numactions(self, a):
        return self.bimatrix.numactions()[a]

    def set_matrix(self, A=None, B=None):
        self.bimatrix = matrix.BiMatrix(A=A, B=B)
        for p in range(2):
            print('matrix[{}]:'.format(p), pprint.pformat(self.bimatrix.matrix()[p]))

    def simulate(self, actions, verbose=False):  # state, actions -> state, reward
        return self.state, np.array([self.bimatrix.get_reward_for(i, actions) for i in range(2)])


class PenaltyShoot(BiMatrixGame):
    def __init__(self, H):
        super().__init__('penaltyshoot', 0.95, H)
        self.set_matrix(
            A=[[-1, 1], [1, -1]])


class RockPaperScissors(BiMatrixGame):
    def __init__(self, H):
        super().__init__('rockpaperscissors', 0.95, H)
        self.set_matrix(
            A=[[0, -1, 1], [1, 0, -1], [-1, 1, 0]])


class PrisonersDilemma(BiMatrixGame):
    def __init__(self, H):
        super().__init__('prisonersdilemma', 0.95, H)
        self.set_matrix(
            A=[[1, 0], [2, 0]],
            B=[[1, 2], [0, 0]])


class PeaceWar(BiMatrixGame):
    def __init__(self, H):
        super().__init__('peacewar', 0.95, H)
        self.set_matrix(
            A=[[2, 0], [3, 1]],
            B=[[2, 3], [0, 1]])


class CrossStreet(BiMatrixGame):
    def __init__(self, H):
        super().__init__('crossstreet', 0.95, H)
        self.set_matrix(
            A=[[1, -1], [-1, 1]],
            B=[[1, -1], [-1, 1]])


class MatchingPennies(BiMatrixGame):
    def __init__(self, H):
        super().__init__('matchingpennies', 0.95, H)
        self.set_matrix(
            A=[[1, -1], [-1, 1]])


class Inspection(BiMatrixGame):
    def __init__(self, H):
        super().__init__('inspection', 0.95, H)
        self.set_matrix(
            A=[[-1, 1], [1, -1]])


class Chicken(BiMatrixGame):
    def __init__(self, H):
        super().__init__('chicken', 0.95, H)
        self.set_matrix(
            A=[[0, 7], [2, 6]],
            B=[[0, 2], [7, 6]])


class RockPaperScissorsSpockLizard(BiMatrixGame):
    def __init__(self, H):
        super().__init__('rockpaperscissorsspocklizard', 0.95, H)
        self.set_matrix(
            A=[[0, -1, 1, -1, 1],
               [1, 0, -1, 1, -1],
               [-1, 1, 0, -1, 1],
               [1, -1, 1, 0, -1],
               [-1, 1, -1, 1, 0]])


class RandomGame(BiMatrixGame):
    def __init__(self, H, n, m, zero_sum=True):
        super().__init__('randomgame', 0.95, H)
        A = RandomGame.random_reward(n, m)
        B = -A if zero_sum else RandomGame.random_reward(n, m)
        self.set_matrix(A=A, B=B)

    @staticmethod
    def random_reward(n, m):
        high = np.full([n, m], 1.0)
        low = np.full([n, m], -1.0)
        return low + (high - low) * np.random.rand(n, m)
