# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import random
import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import partial

import numpy as np

import pickle
import strategy

__author__ = 'Aijun Bai'


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, no, game, name, train=True):
        self.no = no  # player number
        self.name = name
        self.game = game
        self.train = train
        self.numactions = game.numactions(self.no)
        self.opp_numactions = game.numactions(1 - self.no)

    @abstractmethod
    def done(self):
        print('agent {}_{} done...'.format(self.name, self.no))

    @abstractmethod
    def act(self, s):
        pass

    @abstractmethod
    def update(self, s, a, o, r, ns):
        pass

    def pickle_name(self):
        return '{}_{}.pickle'.format(self.name, self.no)


class StationaryAgent(Agent):
    def __init__(self, no, game, pi=None):
        super().__init__(no, game, 'stationary', train=False)
        self.strategy = strategy.Strategy(self.numactions, pi=pi)

    def done(self):
        super().done()

    def act(self, s):
        return self.strategy.sample()

    def update(self, s, a, o, r, ns):
        pass

class RandomAgent(StationaryAgent):
    def __init__(self, no, game):
        n = game.numactions(no)
        super().__init__(no, game, [1.0 / n] * n)


class QAgent(Agent):
    def __init__(self, no, game, train=True, episilon=0.2, alpha=1.0):
        super().__init__(no, game, 'q', train=train)

        self.episilon = episilon
        self.alpha = alpha

        if self.train:
            self.Q = defaultdict(partial(np.random.rand, self.numactions))
        else:
            with open(self.pickle_name(), 'rb') as f:
                self.Q = pickle.load(f)

    def done(self):
        super().done()
        if self.train:
            with open(self.pickle_name(), 'wb') as f:
                pickle.dump(self.Q, f)

    def act(self, s):
        if self.train and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            return np.argmax(self.Q[s])

    def update(self, s, a, o, r, ns):
        self.Q[s][a] += self.alpha * (r + self.game.gamma * np.max(self.Q[ns]) - self.Q[s][a])
        self.alpha *= 0.9999954


class MinimaxQAgent(Agent):
    def __init__(self, no, game, train=True, episilon=0.2, alpha=1.0):
        super().__init__(no, game, 'minimax', train=train)

        self.episilon = episilon
        self.alpha = alpha

        if self.train:
            self.Q = defaultdict(partial(np.random.rand, self.numactions, self.opp_numactions))
            self.strategy = defaultdict(partial(strategy.Strategy, self.numactions))
        else:
            with open(self.pickle_name(), 'rb') as f:
                self.Q, self.strategy = pickle.load(f)

    def done(self):
        super().done()

        if self.train:
            with open(self.pickle_name(), 'wb') as f:
                pickle.dump((self.Q, self.strategy), f)

    def val(self, s):
        return min(np.dot(self.strategy[s].pi, self.Q[s][:, o])
                   for o in range(self.opp_numactions))

    def act(self, s):
        if self.train and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            return self.strategy[s].sample()

    def lp_solve(self, s, solver='scipy'):
        if solver == 'gurobipy':
            import gurobipy as grb
            m = grb.Model('LP')
            m.setParam('OutputFlag', 0)
            m.setParam('LogFile', '')
            m.setParam('LogToConsole', 0)
            v = m.addVar(name='v')
            pi = {}
            for a in range(self.numactions):
                pi[a] = m.addVar(lb=0.0, ub=1.0, name='pi_{}'.format(a))
            m.update()
            m.setObjective(v, sense=grb.GRB.MAXIMIZE)
            for o in range(self.opp_numactions):
                m.addConstr(
                    grb.quicksum(pi[a] * self.Q[s][a, o] for a in range(self.numactions)) >= v,
                    name='c_o{}'.format(o))
            m.addConstr(grb.quicksum(pi[a] for a in range(self.numactions)) == 1, name='c_pi')
            m.optimize()
            return [pi[a].X for a in range(self.numactions)]
        elif solver == 'pulp':
            import pulp as lp
            v = lp.LpVariable('v')
            pi = lp.LpVariable.dicts('pi', list(range(self.numactions)), 0, 1)
            prob = lp.LpProblem('LP', lp.LpMaximize)
            prob += v
            for o in range(self.opp_numactions):
                prob += lp.lpSum(pi[a] * self.Q[s][a, o] for a in range(self.numactions)) >= v
            prob += lp.lpSum(pi[a] for a in range(self.numactions)) == 1
            prob.solve(lp.GLPK_CMD(msg=0))
            return [lp.value(pi[a]) for a in range(self.numactions)]
        elif solver == 'scipy':
            from scipy.optimize import linprog
            c = np.append(np.zeros(self.numactions), -1.0)
            A_ub = np.c_[-self.Q[s].T, np.ones(self.numactions)]
            b_ub = np.zeros(self.numactions)
            A_eq = np.array([np.append(np.ones(self.numactions), 0.0)])
            b_eq = np.array([1.0])
            bounds = [(0.0, 1.0) for _ in range(self.numactions)] + [(-np.inf, np.inf)]
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            return res.x[:-1]

        return None

    def update(self, s, a, o, r, ns):
        self.Q[s][a, o] += self.alpha * (r + self.game.gamma * self.val(ns) - self.Q[s][a, o])
        self.alpha *= 0.9999954

        for solver in ['gurobipy', 'scipy', 'pulp']:
            try:
                self.strategy[s].update(self.lp_solve(s, solver=solver))
            except Exception:
                continue
            else:
                break


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
