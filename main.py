# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import matrix
import agent
import utils
import random
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'Aijun Bai'


def penaltyshoot(gamma=0.95, H=1000):
    A = np.mat('-1, 1; 1, -1')
    B = np.mat('1, -1; -1, 1')
    penalty = matrix.BiMatrix(A, B)

    # attacker = agent.StationaryAgent(0, gamma, 2, [0.9, 0.1])
    # attacker = agent.QAgent(0, gamma, 2, 0.1, 0.01)
    # attacker = agent.MinimaxQAgent(0, gamma, 2, 2, 0.1, 0.01)
    attacker = agent.KappaAgent(0, gamma, 2, 2, 10, 0.1, 0.01)

    goalie = agent.StationaryAgent(1, gamma, 2, [0.9, 0.1])
    # goalie = agent.RandomAgent(1, gamma, 2)
    # goalie = agent.QAgent(1, gamma, 2, 0.1, 0.01)
    # goalie = agent.MinimaxQAgent(1, gamma, 2, 2, 0.1, 0.01)
    # goalie = agent.KappaAgent(1, gamma, 2, 2, 10, 0.1, 0.01)

    learning, policies = [], []
    r = 0.0  # reward for attacker
    for i in xrange(H):
        # print 'learning @ {}'.format(i)
        a = attacker.act(True)
        g = goalie.act(True)
        ra = penalty.get_reward_for(attacker.no, a, g)
        rg = penalty.get_reward_for(goalie.no, a, g)
        attacker.update(a, g, ra)
        goalie.update(g, a, rg)
        r += gamma**i * penalty.get_reward_for(attacker.no, a, g)
        learning.append(r)
        policies.append(attacker.policy())

    attacker.report()
    goalie.report()

    plt.plot(zip(*policies)[0], 'bo-')
    plt.xlabel('t')
    plt.ylabel('policy')
    plt.show()

    return learning

def main():
    learning = utils.makehash()
    N = 1
    H = 10000

    for i in xrange(N):
        learning[i] = penaltyshoot(0.95, H)

    # avg_learning = [utils.mean([learning[i][j] for i in xrange(N)]) for j in xrange(H)]
    # plt.plot(avg_learning, 'r')
    # plt.xlabel('t')
    # plt.ylabel('learning/performance')
    # plt.show()

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    main()