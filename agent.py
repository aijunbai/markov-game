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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import importlib
import humanfriendly
import utils

__author__ = 'Aijun Bai'


class Agent(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, id_, game):
        self.name = name

    def done(self, id_, game):
        pass

    @abstractmethod
    def act(self, s, exploration, id_, game):
        pass

    def update(self, s, a, o, r, sp, id_, game):
        pass

    @staticmethod
    def format_time(n):
        s = humanfriendly.format_size(n)
        return s.replace(' ', '').replace('bytes', '').replace('byte', '').rstrip('B')

    def full_name(self, id_, game):
        return '{}_{}_{}_{}'.format(game.name, self.name, id_, Agent.format_time(game.t))


class StationaryAgent(Agent):
    def __init__(self, id_, game, pi=None):
        super().__init__('stationary', id_, game)
        self.pi = np.array(pi)

    def act(self, s, exploration, id_, game):
        return StationaryAgent.sample(self.pi)

    @staticmethod
    def sample(pi):
        pi /= np.sum(pi)
        return np.random.choice(pi.size, size=1, p=pi)[0]


class RandomAgent(StationaryAgent):
    def __init__(self, id_, game):
        n = game.numactions(id_)
        super().__init__(id_, game, pi=[1.0 / n] * n)
        self.name = 'random'


class BaseQAgent(Agent):
    def __init__(self, name, id_, game, episilon=0.2, N=10000, num_plots=10):
        super().__init__(name, id_, game)
        self.episilon = episilon
        self.N = N
        self.Q = None
        self.pi = {0: defaultdict(partial(np.random.dirichlet, [1.0] * game.numactions(0))),
                   1: defaultdict(partial(np.random.dirichlet, [1.0] * game.numactions(1)))}

        self.num_plots = num_plots
        self.record = defaultdict(list)

    def alpha(self, t):
        return self.N / (self.N + t)

    def plot_record(self, s, record, id_, game):
        fig = plt.figure(figsize=(18,10))
        n = game.numactions(id_)
        for a in range(n):
            plt.subplot(n, 1, a + 1)
            plt.tight_layout()
            plt.gca().set_ylim([-0.1, 1.1])
            plt.title('{}: action {}'.format(self.full_name(id_, game), a))
            plt.xlabel('step')
            plt.ylabel('pi[a]')
            plt.grid()
            x, y = list(zip(*((t, pi[a]) for t, pi in record)))
            plt.plot(x, y, 'r-')
        fig.savefig('policy/' + self.full_name(id_, game) + '.png')
        plt.close(fig)

    def done(self, id_, game):
        super().done(id_, game)

        for s, record in sorted(self.record.items(), key=lambda x: -len(x[1]))[:self.num_plots]:
            self.plot_record(s, record, id_, game)
        del self.record  # prepare for pickling

        if game.verbose:
            utils.pv('self.full_name(id_, game)')
            utils.pv('self.Q')
            utils.pv('self.pi')

    def act(self, s, exploration, id_, game):
        if exploration and random.random() < self.episilon:
            return random.randint(0, game.numactions(id_) - 1)
        else:
            if game.verbose:
                print('policy of {}: {}'.format(id_, self.pi[id_][s]))
            return StationaryAgent.sample(self.pi[id_][s])

    @abstractmethod
    def update(self, s, a, o, r, sp, id_, game):
        pass

    @abstractmethod
    def update_policy(self, s, id_, game):
        if self.record[s]:
            self.record[s].append((game.t - 0.01, self.record[s][-1][1]))
        self.record[s].append((game.t, np.copy(self.pi[id_][s])))

    @abstractmethod
    def do_symmetry(self, s, id_, game):
        pass

class QAgent(BaseQAgent):
    def __init__(self, id_, game):
        super().__init__('q', id_, game)
        self.Q = {0: defaultdict(partial(np.random.rand, game.numactions(0))),
                  1: defaultdict(partial(np.random.rand, game.numactions(1)))}

    def update(self, s, a, o, r, sp, id_, game):
        Q = self.Q[id_][s]
        v = np.max(self.Q[id_][sp])
        Q[a] += self.alpha(game.t) * (r + game.gamma * v - Q[a])
        self.update_policy(s, id_, game)

        if game.is_symmetric:
            self.do_symmetry(s, id_, game)

    def update_policy(self, s, id_, game):
        Q = self.Q[id_][s]
        self.pi[id_][s] = (Q == max(Q)).astype(np.double)
        super().update_policy(s, id_, game)

    def do_symmetry(self, s, id_, game):
        s2 = game.symmetric_state(s)
        for a in range(game.numactions(id_)):
            self.pi[1 - id_][s2][game.symmetric_action(a)] = self.pi[id_][s][a]
            self.Q[1 - id_][s2][game.symmetric_action(a)] = self.Q[id_][s][a]


class MinimaxQAgent(BaseQAgent):
    def __init__(self, id_, game):
        super().__init__('minimax', id_, game)
        self.solvers = []
        self.Q = {0: defaultdict(partial(np.random.rand, game.numactions(0), game.numactions(1))),
                  1: defaultdict(partial(np.random.rand, game.numactions(1), game.numactions(0)))}

    def done(self, id_, game):
        super().done(id_, game)
        del self.solvers  # prepare for pickling

    def val(self, s, id_, game):
        Q = self.Q[id_][s]
        pi = self.pi[id_][s]
        return min(np.dot(pi, Q[:, o]) for o in range(game.numactions(1 - id_)))

    def update(self, s, a, o, r, sp, id_, game):
        Q = self.Q[id_][s]
        v = self.val(sp, id_, game)
        Q[a, o] += self.alpha(game.t) * (r + game.gamma * v - Q[a, o])
        self.update_policy(s, id_, game)

        if game.is_symmetric:
            self.do_symmetry(s, id_, game)

    def update_policy(self, s, id_, game):
        self.initialize_solvers()
        for solver, lib in self.solvers:
            try:
                self.pi[id_][s] = MinimaxQAgent.lp_solve(self.Q[id_][s], solver, lib, id_, game)
            except Exception as e:
                print('optimization using {} failed: {}'.format(solver, e))
                continue
            else:
                break
        super().update_policy(s, id_, game)

    def do_symmetry(self, s, id_, game):
        s2 = game.symmetric_state(s)
        for a in range(game.numactions(id_)):
            self.pi[1 - id_][s2][game.symmetric_action(a)] = self.pi[id_][s][a]
            for o in range(game.numactions(1 - id_)):
                self.Q[1 - id_][s2][game.symmetric_action(a), game.symmetric_action(o)] \
                    = self.Q[id_][s][a, o]

    def initialize_solvers(self):
        if not self.solvers:
            for lib in ['gurobipy', 'scipy.optimize', 'pulp']:
                try:
                    self.solvers.append((lib, importlib.import_module(lib)))
                except:
                    pass

    @staticmethod
    def lp_solve(Q, solver, lib, id_, game):
        ret = None
        numactions =  game.numactions(id_)
        opp_numactions = game.numactions(1 - id_)

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
#     def __init__(self, id_, game, N=25, episilon=0.1, alpha=0.01):
#         super().__init__(id_, game, 'kapper')
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
#         eq = RandomAgent(self.id_, self.game).strategy.pi
#         policies = ({'dist': np.linalg.norm(p.strategy.pi - eq), 'pi': p.strategy, 'val': p.val()} for i, p in
#                     enumerate(self.particles))
#         policies = sorted(policies, key=lambda x: x['dist'])
#         for i, p in enumerate(policies):
#             print('policy_{}:'.format(i), p)
