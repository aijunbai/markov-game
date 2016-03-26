# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import collections

import matplotlib.pyplot as plt
import numpy as np

import matrix

__author__ = 'Aijun Bai'


class Game(object):
    def __init__(self, name, gamma, H):
        self.name = name
        self.gamma = gamma
        self.H = int(H)

        self.bimatrix = None
        self.players = set()

    def add_player(self, player):
        self.players.add(player)

    def update_matrix(self, A, B):
        self.bimatrix = matrix.BiMatrix(np.mat(A), np.mat(B))

    def numactions(self, a):
        return self.bimatrix.numactions()[a]

    def simulate(self):
        policies = collections.defaultdict(list)

        for i in xrange(self.H):
            actions = {player.no(): player.act(True) for player in self.players}
            rewards = {j: self.bimatrix.get_reward_for(j, actions) for j in range(2)}

            for player in self.players:
                j = player.no()
                player.update(actions[j], actions[1 - j], rewards[j])
                policies[j].append(player.policy())

        for player in self.players:
            player.report()

        plt.subplot(211)
        plt.gca().set_xlim([0, 1])
        plt.gca().set_ylim([0, 1])
        plt.title('{}: player {}'.format(self.name, 0))
        plt.xlabel('t')
        plt.ylabel('policy')
        plt.plot(zip(*policies[0])[0], zip(*policies[0])[1], 'ro-')
        plt.subplot(212)
        plt.gca().set_xlim([0, 1])
        plt.gca().set_ylim([0, 1])
        plt.title('{}: player {}'.format(self.name, 1))
        plt.xlabel('t')
        plt.ylabel('policy')
        plt.plot(zip(*policies[1])[0], zip(*policies[1])[1], 'ro-')
        plt.show()


class PenaltyShoot(Game):
    def __init__(self, H):
        super(PenaltyShoot, self).__init__('penaltyshoot', 0.95, H)
        self.update_matrix(
            '-1, 1; 1, -1',
            '1, -1; -1, 1')


class RockPaperScissors(Game):
    def __init__(self, H):
        super(RockPaperScissors, self).__init__('rockpaperscissors', 0.95, H)
        self.update_matrix(
            '0, -1, 1; 1, 0, -1; -1, 1, 0',
            '0, 1, -1; -1, 0, 1; 1, -1, 0')


class PrisonersDilemma(Game):
    def __init__(self, H):
        super(PrisonersDilemma, self).__init__('prisonersdilemma', 0.95, H)
        self.update_matrix(
            '1, 0; 2, 0',
            '1, 2; 0, 0')


class PeaceWar(Game):
    def __init__(self, H):
        super(PeaceWar, self).__init__('peacewar', 0.95, H)
        self.update_matrix(
            '2, 0; 3, 1',
            '2, 3; 0, 1')


class Cross(Game):
    def __init__(self, H):
        super(Cross, self).__init__('cross', 0.95, H)
        self.update_matrix(
            '1, -1; -1, 1',
            '1, -1; -1, 1')
