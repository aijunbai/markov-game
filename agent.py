# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import random
import sys
import os
import math
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
        self.id_ = id_
        self.numactions = game.numactions(id_)
        self.opp_numactions = game.numactions(1 - id_)

    def done(self, game):
        pass

    @abstractmethod
    def act(self, s, exploration, game):
        pass

    def update(self, s, a, o, r, s2, game):
        pass

    @staticmethod
    def format_time(n):
        s = humanfriendly.format_size(n)
        return s.replace(' ', '').replace('bytes', '').replace('byte', '').rstrip('B')

    def full_name(self, game):
        return '{}_{}_{}_{}'.format(game.name, self.name, self.id_, Agent.format_time(game.t))


class StationaryAgent(Agent):
    def __init__(self, id_, game, pi=None):
        super().__init__('stationary', id_, game)
        if pi is None:
            pi = np.random.dirichlet([1.0] * self.numactions)
        self.pi = np.array(pi, dtype=np.double)
        StationaryAgent.normalize(self.pi)

    def act(self, s, exploration, game):
        if game.verbose:
            print('pi of {}: {}'.format(self.id_, self.pi))
        return StationaryAgent.sample(self.pi)

    @staticmethod
    def normalize(pi):
        minprob = np.min(pi)
        if minprob < 0.0:
            pi -= minprob
        pi /= np.sum(pi)

    @staticmethod
    def sample(pi):
        return np.random.choice(pi.size, size=1, p=pi)[0]


class RandomAgent(StationaryAgent):
    def __init__(self, id_, game):
        n = game.numactions(id_)
        super().__init__(id_, game, pi=[1.0 / n] * n)
        self.name = 'random'


class BaseQAgent(Agent):
    def __init__(self, name, id_, game, N=10000, episilon=0.2):
        super().__init__(name, id_, game)
        self.episilon = episilon
        self.N = N
        self.Q = None
        self.pi = defaultdict(partial(np.random.dirichlet, [1.0] * self.numactions))
        self.record = defaultdict(list)

    def done(self, game):
        if game.verbose:
            utils.pv('self.full_name(game)')
            utils.pv('self.Q')
            utils.pv('self.pi')

        numplots = game.numplots if game.numplots >= 0 else len(self.record)
        for s, record in sorted(
                self.record.items(), key=lambda x: -len(x[1]))[:numplots]:
            self.plot_record(s, record, game)
        self.record.clear()


    def alpha(self, t):
        return self.N / (self.N + t)

    def act(self, s, exploration, game):
        if exploration and random.random() < self.episilon:
            return random.randint(0, self.numactions - 1)
        else:
            if game.verbose:
                print('Q of {}: {}'.format(self.id_, self.Q[s]))
                print('pi of {}: {}'.format(self.id_, self.pi[s]))
            return StationaryAgent.sample(self.pi[s])

    @abstractmethod
    def update(self, s, a, o, r, s2, game):
        pass

    @abstractmethod
    def update_policy(self, s, a, game):
        pass

    def plot_record(self, s, record, game):
        os.makedirs('policy/', exist_ok=True)
        fig = plt.figure(figsize=(18, 10))
        n = self.numactions
        for a in range(n):
            plt.subplot(n, 1, a + 1)
            plt.tight_layout()
            plt.gca().set_ylim([-0.05, 1.05])
            plt.gca().set_xlim([1.0, game.t + 1.0])
            plt.title('player: {}: state: {}, action: {}'.format(self.full_name(game), s, a))
            plt.xlabel('step')
            plt.ylabel('pi[a]')
            plt.grid()
            x, y = list(zip(*((t, pi[a]) for t, pi in record)))
            x, y = list(x) + [game.t + 1.0], list(y) + [y[-1]]
            plt.plot(x, y, 'r-')
        fig.savefig('policy/{}_{}.png'.format(self.full_name(game), s))
        plt.close(fig)

    def record_policy(self, s, game):
        if game.numplots != 0:
            if s in self.record:
                self.record[s].append((game.t - 0.01, self.record[s][-1][1]))
            self.record[s].append((game.t, np.copy(self.pi[s])))


class QAgent(BaseQAgent):
    def __init__(self, id_, game):
        super().__init__('q', id_, game)
        self.Q = defaultdict(partial(np.random.rand, self.numactions))
        self.R = defaultdict(partial(np.zeros, self.numactions))
        self.count_R = defaultdict(partial(np.zeros, self.numactions))

    def done(self, game):
        self.R.clear()
        self.count_R.clear()
        super().done(game)

    def update(self, s, a, o, r, s2, game):
        self.count_R[s][a] += 1.0
        self.R[s][a] += (r - self.R[s][a]) / self.count_R[s][a]
        Q = self.Q[s]
        V = self.val(s2)
        Q[a] += self.alpha(game.t) * (self.R[s][a] + game.gamma * V - Q[a])
        self.update_policy(s, a, game)
        self.record_policy(s, game)

    def val(self, s):
        return np.max(self.Q[s])

    def update_policy(self, s, a, game):
        Q = self.Q[s]
        self.pi[s] = (Q == np.max(Q)).astype(np.double)


class PHCAgent(QAgent):
    def __init__(self, game, delta=0.02):
        super().__init__(game)
        self.name = 'phc'
        self.delta = delta

    def update_policy(self, s, a, game):
        delta = self.delta * self.alpha(game.t)
        if a == np.argmax(self.Q[s]):
            self.pi[s][a] += delta
        else:
            self.pi[s][a] -= delta / (self.numactions - 1)
        StationaryAgent.normalize(self.pi[s])


class WoLFAgent(PHCAgent):
    def __init__(self, game, delta1=0.01, delta2=0.04):
        super().__init__(game)
        self.name = 'wolf'
        self.delta1 = delta1
        self.delta2 = delta2
        self.pi_ = defaultdict(partial(np.random.dirichlet, [1.0] * self.numactions))
        self.count_pi = defaultdict(int)

    def done(self, game):
        self.pi_.clear()
        self.count_pi.clear()
        super().done(game)

    def update_policy(self, s, a, game):
        self.count_pi[s] += 1
        self.pi_[s] += (self.pi[s] - self.pi_[s]) / self.count_pi[s]
        self.delta = self.delta1 \
            if np.dot(self.pi[s], self.Q[s]) \
               > np.dot(self.pi_[s], self.Q[s]) \
            else self.delta2
        super().update_policy(s, a, game)


class MinimaxQAgent(BaseQAgent):
    def __init__(self, id_, game):
        super().__init__('minimax', id_, game)
        self.solvers = []
        self.Q = defaultdict(partial(np.random.rand, self.numactions, self.opp_numactions))

    def done(self, game):
        self.solvers.clear()
        super().done(game)

    def val(self, s):
        Q = self.Q[s]
        pi = self.pi[s]
        return min(np.dot(pi, Q[:, o]) for o in range(self.opp_numactions))

    def update(self, s, a, o, r, s2, game):
        Q = self.Q[s]
        V = self.val(s2)
        Q[a, o] += self.alpha(game.t) * (r + game.gamma * V - Q[a, o])
        self.update_policy(s, a, game)
        self.record_policy(s, game)

    def update_policy(self, s, a, game):
        self.initialize_solvers()
        for solver, lib in self.solvers:
            try:
                self.pi[s] = self.lp_solve(self.Q[s], solver, lib)
                StationaryAgent.normalize(self.pi[s])
            except Exception as e:
                print('optimization using {} failed: {}'.format(solver, e))
                continue
            else: break

    def initialize_solvers(self):
        if not self.solvers:
            for lib in ['gurobipy', 'scipy.optimize', 'pulp']:
                try: self.solvers.append((lib, importlib.import_module(lib)))
                except: pass

    def lp_solve(self, Q, solver, lib):
        ret = None

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


class MetaControlAgent(Agent):
    def __init__(self, id_, game):
        super().__init__('metacontrol', id_, game)
        self.agents = [QAgent(id_, game), MinimaxQAgent(id_, game)]
        self.n = np.zeros(len(self.agents))
        self.controller = None

    def act(self, s, exploration, game):
        print([self.val(i, s) for i in range(len(self.agents))])
        self.controller = np.argmax([self.val(i, s) for i in range(len(self.agents))])
        return self.agents[self.controller].act(s, exploration, game)

    def done(self, game):
        for agent in self.agents:
            agent.done(game)

    def val(self, i, s):
        return self.agents[i].val(s)

    def update(self, s, a, o, r, s2, game):
        for agent in self.agents:
            agent.update(s, a, o, r, s2, game)

        self.n[self.controller] += 1
        print('id: {}, n: {} ({}%)'.format(self.id_, self.n, 100.0 * self.n / np.sum(self.n)))


# class KappaAgent(Agent):
#     def __init__(self, game, N=25, episilon=0.1, alpha=0.01):
#         super().__init__(game, 'kapper')
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
#         eq = RandomAgent(self.self.game).strategy.pi
#         policies = ({'dist': np.linalg.norm(p.strategy.pi - eq), 'pi': p.strategy, 'val': p.val()} for i, p in
#                     enumerate(self.particles))
#         policies = sorted(policies, key=lambda x: x['dist'])
#         for i, p in enumerate(policies):
#             print('policy_{}:'.format(i), p)
