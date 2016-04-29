# coding=utf-8


import collections
import pprint

import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod

import matrix

__author__ = 'Aijun Bai'


class Game(object, metaclass=ABCMeta):
    def __init__(self, name, gamma, H):
        self.name = name
        self.gamma = gamma
        self.H = int(H)
        self.players = set()
        self.state = None

    def add_player(self, player):
        self.players.add(player)

    def numactions(self, a):
        return self.bimatrix.numactions()[a]

    def plot(self, policies, plot_iterations=True):
        if plot_iterations:
            for player in self.players:
                plt.figure(player.no + 1)
                for action in range(self.numactions(player.no)):
                    plt.subplot(self.numactions(player.no), 1, action + 1)
                    plt.tight_layout()
                    plt.gca().set_ylim([-0.1, 1.1])
                    plt.title('{}: player {} action {}'.format(self.name, player.no, action))
                    plt.xlabel('iteration')
                    plt.ylabel('probability')
                    plt.grid()
                    plt.plot(list(zip(*policies[player.no]))[action], 'r-')
        else:
            for player in self.players:
                plt.figure(player.no + 1)
                plt.gca().set_xlim([-0.1, 1.1])
                plt.gca().set_ylim([-0.1, 1.1])
                plt.title('{}: player {}'.format(self.name, player.no))
                plt.xlabel('probability')
                plt.ylabel('probability')
                plt.grid()
                plt.plot(list(zip(*policies[player.no]))[0], list(zip(*policies[player.no]))[1], 'ro-')

                circle = plt.Circle((policies[player.no][-1][0], policies[player.no][-1][1]), radius=.02, color='b',
                                    fill=False)
                plt.gca().add_artist(circle)

        plt.show()

    def run(self):
        assert len(self.players) == 2
        policies = collections.defaultdict(list)

        for i in range(self.H):
            actions = {player.no: player.act(True) for player in self.players}
            rewards = self.simulate(actions)

            for player in self.players:
                j = player.no
                player.update(actions[j], actions[1 - j], rewards[j])  # TODO: add states
                policies[j].append(player.policy())

        for player in self.players:
            player.report()

        if self.numactions(0) >= 3 and self.numactions(1) >= 3:
            self.plot(policies, plot_iterations=False)
            # self.plot(policies, plot_iterations=True)
        else:
            self.plot(policies, plot_iterations=True)

    @abstractmethod
    def simulate(self, actions):  # state, actions -> state, reward
        pass
