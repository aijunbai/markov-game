# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import collections
import pprint

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
        print 'matrix[0]:', pprint.pformat(self.bimatrix.matrix()[0])
        print 'matrix[1]:', pprint.pformat(self.bimatrix.matrix()[1])

    def numactions(self, a):
        return self.bimatrix.numactions()[a]

    def plot(self, policies, plot_iterations=True):
        if plot_iterations:
            for player in self.players:
                plt.figure(player.no() + 1)
                for action in range(self.numactions(player.no())):
                    plt.subplot(self.numactions(player.no()), 1, action + 1)
                    plt.tight_layout()
                    plt.gca().set_ylim([-0.1, 1.1])
                    plt.title('{}: player {} action {}'.format(self.name, player.no(), action))
                    plt.xlabel('iteration')
                    plt.ylabel('probability')
                    plt.grid()
                    plt.plot(zip(*policies[player.no()])[action], 'r-')
        else:
            for player in self.players:
                plt.figure(player.no() + 1)
                plt.gca().set_xlim([-0.1, 1.1])
                plt.gca().set_ylim([-0.1, 1.1])
                plt.title('{}: player {}'.format(self.name, player.no()))
                plt.xlabel('probability')
                plt.ylabel('probability')
                plt.grid()
                plt.plot(zip(*policies[player.no()])[0], zip(*policies[player.no()])[1], 'ro-')

                circle = plt.Circle((policies[player.no()][-1][0], policies[player.no()][-1][1]), radius=.02, color='b',
                                    fill=False)
                plt.gca().add_artist(circle)

        plt.show()

    def simulate(self):
        assert len(self.players) == 2
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

        if self.numactions(0) == 3 and self.numactions(1) == 3:
            self.plot(policies, plot_iterations=True)
            self.plot(policies, plot_iterations=False)
        else:
            self.plot(policies, plot_iterations=True)


class PenaltyShoot(Game):
    def __init__(self, H):
        super(PenaltyShoot, self).__init__('penaltyshoot', 0.95, H)
        self.update_matrix(
            A='-1, 1; 1, -1',
            B='1, -1; -1, 1'
        )


class RockPaperScissors(Game):
    def __init__(self, H):
        super(RockPaperScissors, self).__init__('rockpaperscissors', 0.95, H)
        self.update_matrix(
            A='0, -1, 1; 1, 0, -1; -1, 1, 0',
            B='0, 1, -1; -1, 0, 1; 1, -1, 0'
        )


class PrisonersDilemma(Game):
    def __init__(self, H):
        super(PrisonersDilemma, self).__init__('prisonersdilemma', 0.95, H)
        self.update_matrix(
            A='1, 0; 2, 0',
            B='1, 2; 0, 0'
        )


class PeaceWar(Game):
    def __init__(self, H):
        super(PeaceWar, self).__init__('peacewar', 0.95, H)
        self.update_matrix(
            A='2, 0; 3, 1',
            B='2, 3; 0, 1'
        )


class CrossStreet(Game):
    def __init__(self, H):
        super(CrossStreet, self).__init__('crossstreet', 0.95, H)
        self.update_matrix(
            A='1, -1; -1, 1',
            B='1, -1; -1, 1'
        )


class MatchingPennies(Game):
    def __init__(self, H):
        super(MatchingPennies, self).__init__('matchingpennies', 0.95, H)
        self.update_matrix(
            A='1, -1; -1, 1',
            B='-1, 1; 1, -1'
        )


class Inspection(Game):
    def __init__(self, H, c):
        super(Inspection, self).__init__('inspection', 0.95, H)
        self.update_matrix(
            A='-1, 1; 1, -1',
            B='1, -1; -1, 1'
        )
