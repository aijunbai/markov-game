# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import math
import random

import numpy as np

import agent
import game

__author__ = 'Aijun Bai'

seed = 0


def experiment(H):
    # g = game.PenaltyShoot(H)
    g = game.RockPaperScissors(H)
    # g = game.PrisonersDilemma(H)
    # g = game.PeaceWar(H)
    # g = game.Cross(H)

    # g.add_player(agent.KappaAgent(0, g, N=10))
    # g.add_player(agent.QAgent(0, g))
    g.add_player(agent.MinimaxQAgent(0, g))
    # g.add_player(agent.StationaryAgent(1, g))
    # g.add_player(agent.RandomAgent(1, g))
    # g.add_player(agent.QAgent(1, g))
    # g.add_player(agent.KappaAgent(1, g, N=2))
    g.add_player(agent.MinimaxQAgent(1, g))

    g.simulate()


def main(N=1):
    for i in xrange(N):
        experiment(H=math.pow(2, 10))

if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)

    main(N=1)
