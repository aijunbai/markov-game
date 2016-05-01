# coding=utf-8

import math

import agent
import bimatrixgame
import littmansoccer
import utils

__author__ = 'Aijun Bai'

seed = 0


def experiment(H):
    # g = game.PenaltyShoot(H)
    # g = bimatrixgame.RockPaperScissors(H)
    g = littmansoccer.LittmanSoccer(H)
    # g = game.RockPaperScissorsSpockLizard(H)
    # g = game.PrisonersDilemma(H)
    # g = game.PeaceWar(H)
    # g = game.CrossStreet(H)
    # g = game.MatchingPennies(H)
    # g = game.Inspection(H)
    # g = game.Chicken(H)
    # g = game.RandomGame(H, 2, 2, zero_sum=True)

    # g.add_player(agent.RandomAgent(0, g))
    # g.add_player(agent.QAgent(0, g))
    g.add_player(agent.MinimaxQAgent(0, g))

    # g.add_player(agent.RandomAgent(1, g))
    # g.add_player(agent.QAgent(1, g))
    g.add_player(agent.MinimaxQAgent(1, g))

    g.run(verbose=False)


def main(N):
    for i in range(N):
        experiment(H=math.pow(2, 16))

if __name__ == '__main__':
    if seed is not None:
        utils.random_seed(seed)

    main(N=1)
