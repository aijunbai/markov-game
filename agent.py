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
import humanfriendly
import utils
import strategy

__author__ = 'Aijun Bai'


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, no, game):
        self.name = name
        print('{}: creating agent {}_{}...'.format(game, self.name, no))

    def done(self, no, game):
        print('{}: finishing agent {}_{}...'.format(game, self.name, no))

    @abstractmethod
    def act(self, s, exploration, no, game):
        pass

    def update(self, s, a, o, r, sp, no, game):
        pass

    @staticmethod
    def format(n):
        s = humanfriendly.format_size(n)
        return s.replace(' ', '').replace('bytes', '').replace('byte', '').rstrip('B')

    def pickle_name(self, no, game):
        return 'data/{}_{}_{}_{}.pickle'.format(
            game.name, self.name, no, Agent.format(game.H))


class StationaryAgent(Agent):
    def __init__(self, no, game, pi=None):
        super().__init__('stationary', no, game)
        self.strategy = strategy.Strategy(game.numactions(no), pi=pi)

    def act(self, s, exploration, no, game):
        return self.strategy.sample()


class RandomAgent(StationaryAgent):
    def __init__(self, no, game):
        n = game.numactions(no)
        super().__init__(no, game, pi=[1.0 / n] * n)
        self.name = 'random'


class BaseQAgent(Agent):
    def __init__(self, name, no, game, episilon=0.2, N=10000):
        super().__init__(name, no, game)
        self.episilon = episilon
        self.N = N
        self.Q = None
        self.strategy = {0: defaultdict(partial(strategy.Strategy, game.numactions(0))),
                         1: defaultdict(partial(strategy.Strategy, game.numactions(1)))}

    def alpha(self, t):
        return self.N / (self.N + t)

    def done(self, no, game):
        super().done(no, game)

        if game.verbose:
            utils.pv('self.name')
            utils.pv('self.Q')
            utils.pv('self.strategy')

    def act(self, s, exploration, no, game):
        if exploration and random.random() < self.episilon:
            return random.randint(0, game.numactions(no) - 1)
        else:
            if game.verbose:
                print('strategy of {}: {}'.format(no, self.strategy[no][s]))
            return self.strategy[no][s].sample()

    @abstractmethod
    def update(self, s, a, o, r, sp, no, game):
        pass

    @abstractmethod
    def update_strategy(self, s, no, game):
        pass

    @abstractmethod
    def do_symmetry(self, s, no, game):
        pass

class QAgent(BaseQAgent):
    def __init__(self, no, game):
        super().__init__('q', no, game)
        self.Q = {0: defaultdict(partial(np.random.rand, game.numactions(0))),
                  1: defaultdict(partial(np.random.rand, game.numactions(1)))}

    def update(self, s, a, o, r, sp, no, game):
        Q = self.Q[no][s]
        v = np.max(self.Q[no][sp])
        Q[a] += self.alpha(game.t) * (r + game.gamma * v - Q[a])
        self.update_strategy(s, no, game)

        if game.is_symmetric:
            self.do_symmetry(s, no, game)

    def update_strategy(self, s, no, game):
        Q = self.Q[no][s]
        self.strategy[no][s].update((Q == max(Q)).astype(np.double))

    def do_symmetry(self, s, no, game):
        s2 = game.symmetric_state(s)
        for a in range(game.numactions(no)):
            self.strategy[1 - no][s2].pi[game.symmetric_action(a)] = self.strategy[no][s].pi[a]
            self.Q[1 - no][s2][game.symmetric_action(a)] = self.Q[no][s][a]


class MinimaxQAgent(BaseQAgent):
    def __init__(self, no, game):
        super().__init__('minimax', no, game)
        self.solvers = []
        self.Q = {0: defaultdict(partial(np.random.rand, game.numactions(0), game.numactions(1))),
                  1: defaultdict(partial(np.random.rand, game.numactions(1), game.numactions(0)))}

    def done(self, no, game):
        super().done(no, game)
        self.solvers = []  # preparing for pickling

    def val(self, s, no, game):
        Q = self.Q[no][s]
        pi = self.strategy[no][s].pi
        return min(np.dot(pi, Q[:, o]) for o in range(game.numactions(1 - no)))

    def update(self, s, a, o, r, sp, no, game):
        Q = self.Q[no][s]
        v = self.val(sp, no, game)
        Q[a, o] += self.alpha(game.t) * (r + game.gamma * v - Q[a, o])
        self.update_strategy(s, no, game)

        if game.is_symmetric:
            self.do_symmetry(s, no, game)

    def update_strategy(self, s, no, game):
        self.initialize_solvers()
        for solver, lib in self.solvers:
            try:
                self.strategy[no][s].update(
                    MinimaxQAgent.lp_solve(self.Q[no][s], solver, lib, no, game))
            except Exception as e:
                print('optimization using {} failed: {}'.format(solver, e))
                continue
            else:
                break

    def do_symmetry(self, s, no, game):
        s2 = game.symmetric_state(s)
        for a in range(game.numactions(no)):
            self.strategy[1 - no][s2].pi[game.symmetric_action(a)] = self.strategy[no][s].pi[a]
            for o in range(game.numactions(1 - no)):
                self.Q[1 - no][s2][game.symmetric_action(a), game.symmetric_action(o)] \
                    = self.Q[no][s][a, o]

    def initialize_solvers(self):
        if not self.solvers:
            for lib in ['gurobipy', 'scipy.optimize', 'pulp']:
                try:
                    self.solvers.append((lib, importlib.import_module(lib)))
                except:
                    pass

    @staticmethod
    def lp_solve(Q, solver, lib, no, game):
        ret = None
        numactions =  game.numactions(no)
        opp_numactions = game.numactions(1 - no)

        if solver == 'scipy.optimize':
            c = np.append(np.zeros(numactions), -1.0)
            A_ub = np.c_[-Q.T, np.ones(opp_numactions)]
            b_ub = np.zeros(opp_numactions)
            A_eq = np.array([np.append(np.ones(numactions), 0.0)])
            b_eq = np.array([1.0])
            bounds = [(0.0, 1.0) for _ in range(numactions)] + [(-np.inf, np.inf)]
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
            for a in range(numactions):
                pi[a] = m.addVar(lb=0.0, ub=1.0, name='pi_{}'.format(a))
            m.update()
            m.setObjective(v, sense=lib.GRB.MAXIMIZE)
            for o in range(opp_numactions):
                m.addConstr(
                    lib.quicksum(pi[a] * Q[a, o] for a in range(numactions)) >= v,
                    name='c_o{}'.format(o))
            m.addConstr(lib.quicksum(pi[a] for a in range(numactions)) == 1, name='c_pi')
            m.optimize()
            ret = np.array([pi[a].X for a in range(numactions)])
        elif solver == 'pulp':
            v = lib.LpVariable('v')
            pi = lib.LpVariable.dicts('pi', list(range(numactions)), 0, 1)
            prob = lib.LpProblem('LP', lib.LpMaximize)
            prob += v
            for o in range(opp_numactions):
                prob += lib.lpSum(pi[a] * Q[a, o] for a in range(numactions)) >= v
            prob += lib.lpSum(pi[a] for a in range(numactions)) == 1
            prob.solve(lib.GLPK_CMD(msg=0))
            ret = np.array([lib.value(pi[a]) for a in range(numactions)])

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
