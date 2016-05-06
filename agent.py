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

import importlib
import utils
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
    def update(self, s, a, o, r, sp):
        pass

    def pickle_name(self):
        return 'data/{}_{}_{}.pickle'.format(self.game.name, self.name, self.no)


class StationaryAgent(Agent):
    def __init__(self, no, game, train=False, pi=None):
        super().__init__(no, game, 'stationary', train=train)
        self.strategy = strategy.Strategy(self.numactions, pi=pi)

    def done(self):
        super().done()

    def act(self, s):
        return self.strategy.sample()

    def update(self, s, a, o, r, sp):
        pass

class RandomAgent(StationaryAgent):
    def __init__(self, no, game, train=False):
        n = game.numactions(no)
        super().__init__(no, game, train=train, pi=[1.0 / n] * n)
        self.name = 'random'

class BaseQAgent(Agent):
    def __init__(self, no, game, name, train=True, episilon=0.2, N=10000):
        super().__init__(no, game, name, train=train)
        self.episilon = episilon
        self.N = N
        self.step = 0
        self.Q = None
        self.strategy = defaultdict(partial(strategy.Strategy, self.numactions))

        if not self.train:
            self.load()

    def alpha(self):
        self.step += 1
        return self.N / (self.N + self.step)

    def load(self):
        with open(self.pickle_name(), 'rb') as f:
            self.Q, self.strategy = pickle.load(f)

    def save(self):
        with open(self.pickle_name(), 'wb') as f:
            pickle.dump((self.Q, self.strategy), f, protocol=2)

    def done(self):
        super().done()

        if self.game.verbose:
            utils.pv('self.pickle_name()')
            utils.pv('self.Q')
            utils.pv('self.strategy')

        if self.train:
            self.save()

    def act(self, s):
        if self.train and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            return self.strategy[s].sample()

    @abstractmethod
    def update(self, s, a, o, r, sp):
        pass

    @abstractmethod
    def update_strategy(self, s):
        pass

class QAgent(BaseQAgent):
    def __init__(self, no, game, train=True, episilon=0.2):
        super().__init__(no, game, 'q', train=train, episilon=episilon)
        self.Q = defaultdict(partial(np.random.rand, self.numactions))

    def update(self, s, a, o, r, sp):
        Q = self.Q[s]
        v = np.max(self.Q[sp])
        Q[a] += self.alpha() * (r + self.game.gamma * v - Q[a])
        self.update_strategy(s)

    def update_strategy(self, s):
        Q = self.Q[s]
        self.strategy[s].update((Q == max(Q)).astype(np.double))


class MinimaxQAgent(BaseQAgent):
    def __init__(self, no, game, train=True, episilon=0.2):
        super().__init__(no, game, 'minimax', train=train, episilon=episilon)
        self.Q = defaultdict(partial(np.random.rand, self.numactions, self.opp_numactions))

        self.solvers = []
        for lib in ['gurobipy', 'scipy.optimize', 'pulp']:
            try:
                self.solvers.append((lib, importlib.import_module(lib)))
            except:
                pass

    def val(self, s):
        Q = self.Q[s]
        pi = self.strategy[s].pi
        return min(np.dot(pi, Q[:, o]) for o in range(self.opp_numactions))

    def update(self, s, a, o, r, sp):
        Q = self.Q[s]
        v = self.val(sp)
        Q[a, o] += self.alpha() * (r + self.game.gamma * v - Q[a, o])
        self.update_strategy(s)

    def update_strategy(self, s):
        for solver, lib in self.solvers:
            try:
                self.strategy[s].update(self.lp_solve(self.Q[s], solver, lib))
            except Exception as e:
                print('optimization using {} failed: '.format(solver, e))
                continue
            else:
                break

    def lp_solve(self, Q, solver, lib):
        if solver == 'scipy.optimize':
            c = np.append(np.zeros(self.numactions), -1.0)
            A_ub = np.c_[-Q.T, np.ones(self.opp_numactions)]
            b_ub = np.zeros(self.opp_numactions)
            A_eq = np.array([np.append(np.ones(self.numactions), 0.0)])
            b_eq = np.array([1.0])
            bounds = [(0.0, 1.0) for _ in range(self.numactions)] + [(-np.inf, np.inf)]
            res = lib.linprog(
                c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            ret = res.x[:-1]
        elif solver == 'gurobipy':
            m = lib.Model('LP')
            m.setParam('OutputFlag', 0)
            m.setParam('LogFile', '')
            m.setParam('LogToConsole', 0)
            v = m.addVar(name='v')
            pi = {}
            for a in range(self.numactions):
                pi[a] = m.addVar(lb=0.0, ub=1.0, name='pi_{}'.format(a))
            m.update()
            m.setObjective(v, sense=lib.GRB.MAXIMIZE)
            for o in range(self.opp_numactions):
                m.addConstr(
                    lib.quicksum(pi[a] * Q[a, o] for a in range(self.numactions)) >= v,
                    name='c_o{}'.format(o))
            m.addConstr(lib.quicksum(pi[a] for a in range(self.numactions)) == 1, name='c_pi')
            m.optimize()
            ret = np.array([pi[a].X for a in range(self.numactions)])
        elif solver == 'pulp':
            v = lib.LpVariable('v')
            pi = lib.LpVariable.dicts('pi', list(range(self.numactions)), 0, 1)
            prob = lib.LpProblem('LP', lib.LpMaximize)
            prob += v
            for o in range(self.opp_numactions):
                prob += lib.lpSum(pi[a] * Q[a, o] for a in range(self.numactions)) >= v
            prob += lib.lpSum(pi[a] for a in range(self.numactions)) == 1
            prob.solve(lib.GLPK_CMD(msg=0))
            ret = np.array([lib.value(pi[a]) for a in range(self.numactions)])

        if not (ret >= 0.0).all():
            raise Exception('{} - negative probability error: {}'.format(solver, ret))

        return ret


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
