# coding=utf-8

import collections
import random
from abc import ABCMeta, abstractmethod

import numpy as np
import pulp as lp

import strategy
import utils

__author__ = 'Aijun Bai'


class Agent(object, metaclass=ABCMeta):
    def __init__(self, no, game, name):
        self.no = no  # player number
        self.name = name
        self.game = game
        self.numactions = game.numactions(self.no)
        self.opp_numactions = game.numactions(1 - self.no)

    @abstractmethod
    def act(self, s, exploration):
        pass

    @abstractmethod
    def update(self, s, a, o, r, ns):
        pass


class StationaryAgent(Agent):
    def __init__(self, no, game, pi=None):
        super().__init__(no, game, 'stationary')
        self.strategy = strategy.Strategy(self.numactions, pi)

    def act(self, s, exploration):
        return self.strategy.sample()

    def update(self, s, a, o, r, ns):
        pass

class RandomAgent(StationaryAgent):
    def __init__(self, no, game):
        n = game.numactions(no)
        super().__init__(no, game, [1.0 / n] * n)


class QAgent(Agent):
    def __init__(self, no, game, episilon=0.2, alpha=1.0):
        super().__init__(no, game, 'q')

        self.episilon = episilon
        self.alpha = alpha
        self.Q = collections.defaultdict(lambda: np.random.rand(self.numactions))

    # def __del__(self):
    #     utils.pv('self.Q')

    def act(self, s, exploration):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            return np.argmax(self.Q[s])

    def update(self, s, a, o, r, ns):
        val = max(self.Q[ns][a] for a in range(self.numactions))
        self.Q[s][a] += self.alpha * (r + self.game.gamma * val - self.Q[s][a])
        self.alpha *= 0.9999954


class MinimaxQAgent(Agent):
    def __init__(self, no, game, episilon=0.2, alpha=1.0):
        super().__init__(no, game, 'minimax')

        self.episilon = episilon
        self.alpha = alpha

        self.Q = collections.defaultdict(lambda: np.random.rand(self.numactions, self.opp_numactions))
        self.strategy = collections.defaultdict(lambda: strategy.Strategy(self.numactions))

    # def __del__(self):
    #     utils.pv('self.Q')
    #     utils.pv('self.strategy')

    def val(self, s):
        return min(
            sum(p * q for p, q in zip(
                self.strategy[s].pi, (self.Q[s][a, o] for a in range(self.numactions))))
            for o in range(self.opp_numactions))

    def act(self, s, exploration):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            return self.strategy[s].sample()

    def update(self, s, a, o, r, ns):
        val = self.val(ns)
        self.Q[s][a, o] += self.alpha * (r + self.game.gamma * val - self.Q[s][a, o])
        self.alpha *= 0.9999954

        # update strategy
        v = lp.LpVariable('v')
        pi = lp.LpVariable.dicts('pi', list(range(self.numactions)), 0, 1)
        prob = lp.LpProblem('maximizing', lp.LpMaximize)
        prob += v
        for o in range(self.opp_numactions):
            prob += lp.lpSum(pi[a] * self.Q[s][a, o] for a in range(self.numactions)) >= v
        prob += lp.lpSum(pi[a] for a in range(self.numactions)) == 1
        status = prob.solve(lp.GLPK_CMD(msg=0))

        if status == 1:
            self.strategy[s].update([lp.value(pi[a]) for a in range(self.numactions)])
        else:
            assert 0

# class KappaAgent(Agent):
#     def __init__(self, no, game, N=25, episilon=0.1, alpha=0.01):
#         super().__init__(no, game, 'kapper')
#
#         self.episilon = episilon
#         self.alpha = alpha
#         self.numstrategies = N
#         self.particles = [particle.Particle(self) for _ in range(self.numstrategies)]
#         self.strategy = None
#         self.particles[-1].strategy.pi = np.full(self.particles[-1].strategy.pi.shape, 1.0 / self.numactions)
#
#     def act(self, exploration):
#         if exploration and random.random() < self.episilon:
#             self.strategy = random.choice(self.particles).strategy
#         else:
#             self.strategy = max(self.particles, key=(lambda x: x.val())).strategy
#         return self.strategy.sample()
#
#     def update(self, a, o, r):
#         k = r + self.game.gamma * max(x.val() for x in self.particles)
#
#         total_weight = 0.0
#         for p in self.particles:
#             w = p.strategy.pi[a] / self.strategy.pi[a]
#             p.K[o] += self.alpha * w * (k - p.K[o])
#             total_weight += p.strategy.pi[a]  # weight conditioned on observations?
#
#         distribution = [p.strategy.pi[a] / total_weight for p in self.particles]
#         outcomes = np.random.multinomial(self.numstrategies, distribution)
#         self.particles = [self.particles[i].clone() for i, c in enumerate(outcomes) for _ in range(c)]
#
#     def policy(self):
#         return max(self.particles, key=(lambda x: x.val())).strategy.pi
#
#     def report(self):
#         super().report()
#         eq = RandomAgent(self.no, self.game).strategy.pi
#         policies = ({'dist': np.linalg.norm(p.strategy.pi - eq), 'pi': p.strategy, 'val': p.val()} for i, p in
#                     enumerate(self.particles))
#         policies = sorted(policies, key=lambda x: x['dist'])
#         for i, p in enumerate(policies):
#             print('policy_{}:'.format(i), p)
