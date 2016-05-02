# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

import agent
import littmansoccer
import utils

__author__ = 'Aijun Bai'

seed = 0


def experiment(H):
    # g = bimatrixgame.PenaltyShoot(H)
    # g = bimatrixgame.RockPaperScissors(H)
    g = littmansoccer.LittmanSoccer(H)
    # g = bimatrixgame.RockPaperScissorsSpockLizard(H)
    # g = bimatrixgame.PrisonersDilemma(H)
    # g = bimatrixgame.PeaceWar(H)
    # g = bimatrixgame.CrossStreet(H)
    # g = bimatrixgame.MatchingPennies(H)
    # g = bimatrixgame.Inspection(H)
    # g = bimatrixgame.Chicken(H)
    # g = bimatrixgame.RandomGame(H, 2, 2, zero_sum=True)

    # g.add_player(agent.RandomAgent(0, g))
    g.add_player(agent.QAgent(0, g, train=True))
    # g.add_player(agent.MinimaxQAgent(0, g, train=True))

    # g.add_player(agent.RandomAgent(1, g))
    g.add_player(agent.QAgent(1, g, train=True))
    # g.add_player(agent.MinimaxQAgent(1, g, train=True))

    g.run(verbose=False)

def main(N):
    for i in range(N):
        experiment(H=100000)

if __name__ == '__main__':
    if seed is not None:
        utils.random_seed(seed)

    main(N=1)
