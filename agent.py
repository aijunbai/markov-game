# coding=utf-8

from __future__ import division
from __future__ import with_statement  # for python 2.5

import random
import utils
import strategy
import pprint
import pulp as lp
import numpy as np

__author__ = 'Aijun Bai'


class Agent(object):
    def __init__(self, no, gamma, n):
        self.no = no  # player number
        self.numactions = n
        self.gamma = gamma

    def act(self, exploration):
        pass

    def update(self, a, o, r):
        pass

    def policy(self):
        pass

    def report(self):
        pass


class StationaryAgent(Agent):
    def __init__(self, no, gamma, n, pi=None):
        super(StationaryAgent, self).__init__(no, gamma, n)
        self.strategy = strategy.Strategy(n, pi)

    def act(self, exploration):
        return self.strategy.sample()

    def policy(self):
        return self.strategy.pi()

class RandomAgent(StationaryAgent):
    def __init__(self, no, gamma, n):
        super(RandomAgent, self).__init__(no, gamma, n, [1.0 / n] * n)


class QAgent(Agent):
    def __init__(self, no, gamma, n, episilon=0.1, alpha=0.01):
        super(QAgent, self).__init__(no, gamma, n)

        self.episilon = episilon
        self.alpha = alpha

        self.Q = np.random.rand(self.numactions)

    def act(self, exploration):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            return np.argmax(self.Q)

    def update(self, a, o, r):
        self.Q[a] += self.alpha * (r + self.gamma * max(self.Q[a] for a in range(self.numactions)) - self.Q[a])

    def policy(self):
        distri = [0, 0]
        distri[np.argmax(self.Q)] = 1
        return distri

class MinimaxQAgent(Agent):
    def __init__(self, no, gamma, n, m, episilon=0.1, alpha=0.01):
        super(MinimaxQAgent, self).__init__(no, gamma, n)

        self.episilon = episilon
        self.alpha = alpha
        self.opp_numactions = m

        self.Q = np.random.rand(self.numactions, self.opp_numactions)
        self.strategy = strategy.Strategy(self.numactions)

    def val(self, pi):
        return min(
            sum(p * q for p, q in zip(pi, (self.Q[i][o] for i in xrange(self.numactions))))
            for o in range(self.numactions))

    def act(self, exploration):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            return self.strategy.sample()

    def update(self, a, o, r):
        self.Q[a, o] += self.alpha * (r + self.gamma * self.val(self.strategy.pi()) - self.Q[a, o])

        # update pi
        v = lp.LpVariable('v')
        pi = lp.LpVariable.dicts('pi', range(self.numactions), 0, 1)
        prob = lp.LpProblem('maximizing', lp.LpMaximize)

        prob += v

        for o in range(self.opp_numactions):
            prob += lp.lpSum(pi[i] * self.Q[i, o] for i in range(self.numactions)) >= v

        prob += lp.lpSum(pi[i] for i in range(self.numactions)) == 1

        for i in range(self.numactions):
            prob += pi[i] >= 0
            prob += pi[i] <= 1

        # status = prob.solve(lp.GLPK(msg=False))
        status = prob.solve(lp.COIN())
        # status = prob.solve(lp.GUROBI())
        if status == 1:
            self.strategy.update([lp.value(pi[i]) for i in range(self.numactions)])
        else:
            assert 0

    def policy(self):
        return self.strategy.pi()

    def report(self):
        print 'strategy:', self.strategy
        print 'val:', self.val(self.strategy.pi())
        print 'Q:', self.Q

class KappaAgent(Agent):  # there should be more updates for each policy, and more updates for the set of samples
    # updates for particles -- posterior distributions
    # importance sampling -- I have to work this out!
    def __init__(self, no, gamma, n, m, N, episilon=0.1, alpha=0.01):
        super(KappaAgent, self).__init__(no, gamma, n)

        self.episilon = episilon
        self.alpha = alpha
        self.opp_numactions = m
        self.numstrategies = N

        self.strategies = [strategy.Strategy(self.numactions) for _ in range(self.numstrategies)]
        self.strategies[0] = RandomAgent(no, gamma, n).strategy

        self.K = {s: np.random.rand(self.opp_numactions) for s in self.strategies}
        self.numupdates = {s: np.zeros(self.opp_numactions) for s in self.strategies}
        self.strategy = None

    def val(self, s):
        return min(self.K[s][o] for o in range(self.opp_numactions))

    def act(self, exploration):
        if exploration and random.random() < self.episilon:
            self.strategy = random.choice(self.strategies)
        else:
            self.strategy = max(self.strategies, key=(lambda x: self.val(x)))
        return self.strategy.sample()

    def update(self, a, o, r):
        self.K[self.strategy][o] += self.alpha * (
            r + self.gamma * max(self.val(x) for x in self.strategies) - self.K[self.strategy][o])
        self.numupdates[self.strategy][o] += 1

    def policy(self):
        return max(self.strategies, key=(lambda x: self.val(x))).pi()

    def report(self):
        print 'strategy:', self.strategy
        print 'K:', pprint.pformat(self.K)
        for i, s in enumerate(self.strategies):
            print 'policy:', i, s, self.val(s), sum(self.numupdates[s][o] for o in range(self.opp_numactions))
