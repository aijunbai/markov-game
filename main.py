# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import math

import agent
import game
import utils

__author__ = 'Aijun Bai'

seed = 0


def experiment(H):
    # g = game.PenaltyShoot(H)
    g = game.RockPaperScissors(H)
    # g = game.RockPaperScissorsSpockLizard(H)
    # g = game.PrisonersDilemma(H)
    # g = game.PeaceWar(H)
    # g = game.CrossStreet(H)
    # g = game.MatchingPennies(H)
    # g = game.Inspection(H)
    # g = game.Chicken(H)
    # g = game.RandomGame(H, 2, 2, zero_sum=True)

    g.add_player(agent.KappaAgent(0, g, N=15))
    # g.add_player(agent.QAgent(0, g))
    # g.add_player(agent.MinimaxQAgent(0, g))

    # g.add_player(agent.StationaryAgent(1, g))
    # g.add_player(agent.RandomAgent(1, g))
    g.add_player(agent.QAgent(1, g))
    # g.add_player(agent.KappaAgent(1, g))
    # g.add_player(agent.MinimaxQAgent(1, g))

    g.simulate()


def main(N):
    for i in xrange(N):
        experiment(H=math.pow(2, 10))

if __name__ == '__main__':
    if seed is not None:
        utils.random_seed(seed)

    main(N=1)
