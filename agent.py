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

    def __init__(self, name):
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
        super().__init__('stationary')
        if pi is None:
            pi = np.random.dirichlet([1.0] * game.numactions(id_))
        self.pi = np.array(pi, dtype=np.double)
        StationaryAgent.normalize(self.pi)

    def act(self, s, exploration, id_, game):
        if game.verbose:
            print('pi of {}: {}'.format(id_, self.pi))
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
        super().__init__(name)
        self.episilon = episilon
        self.N = N
        self.Q = None
        self.pi = np.array(
            [defaultdict(partial(np.random.dirichlet, [1.0] * game.numactions(0))),
             defaultdict(partial(np.random.dirichlet, [1.0] * game.numactions(1)))])
        self.record = defaultdict(list)

    def done(self, id_, game):
        if game.verbose:
            utils.pv('self.full_name(id_, game)')
            utils.pv('self.Q')
            utils.pv('self.pi')

        numplots = game.numplots if game.numplots >= 0 else len(self.record)
        for s, record in sorted(
                self.record.items(), key=lambda x: -len(x[1]))[:numplots]:
            self.plot_record(s, record, id_, game)
        self.record.clear()


    def alpha(self, t):
        return self.N / (self.N + t)

    def act(self, s, exploration, id_, game):
        if exploration and random.random() < self.episilon:
            return random.randint(0, game.numactions(id_) - 1)
        else:
            if game.verbose:
                print('Q of {}: {}'.format(id_, self.Q[id_][s]))
                print('pi of {}: {}'.format(id_, self.pi[id_][s]))
            return StationaryAgent.sample(self.pi[id_][s])

    @abstractmethod
    def update(self, s, a, o, r, sp, id_, game):
        pass

    @abstractmethod
    def update_policy(self, s, a, id_, game):
        pass

    def plot_record(self, s, record, id_, game):
        os.makedirs('policy/', exist_ok=True)
        fig = plt.figure(figsize=(18, 10))
        n = game.numactions(id_)
        for a in range(n):
            plt.subplot(n, 1, a + 1)
            plt.tight_layout()
            plt.gca().set_ylim([-0.05, 1.05])
            plt.gca().set_xlim([1.0, game.t + 1.0])
            plt.title('player: {}: state: {}, action: {}'.format(self.full_name(id_, game), s, a))
            plt.xlabel('step')
            plt.ylabel('pi[a]')
            plt.grid()
            x, y = list(zip(*((t, pi[a]) for t, pi in record)))
            x, y = list(x) + [game.t + 1.0], list(y) + [y[-1]]
            plt.plot(x, y, 'r-')
        fig.savefig('policy/{}_{}.png'.format(self.full_name(id_, game), s))
        plt.close(fig)

    def record_policy(self, s, id_, game):
        if game.numplots != 0:
            if s in self.record:
                self.record[s].append((game.t - 0.01, self.record[s][-1][1]))
            self.record[s].append((game.t, np.copy(self.pi[id_][s])))

    @abstractmethod
    def do_symmetry(self, s, id_, game):
        pass


class QAgent(BaseQAgent):
    def __init__(self, id_, game):
        super().__init__('q', id_, game)
        self.R = defaultdict(partial(np.zeros, game.numactions(id_)))  # expected R(s, a)
        self.count_R = defaultdict(partial(np.zeros, game.numactions(id_)))
        self.Q = np.array(
            [defaultdict(partial(np.random.rand, game.numactions(0))),
             defaultdict(partial(np.random.rand, game.numactions(1)))])

    def done(self, id_, game):
        self.R.clear()
        self.count_R.clear()
        super().done(id_, game)

    def update(self, s, a, o, r, sp, id_, game):
        self.count_R[s][a] += 1.0
        self.R[s][a] += (r - self.R[s][a]) / self.count_R[s][a]
        Q = self.Q[id_][s]
        V = np.max(self.Q[id_][sp])
        Q[a] += self.alpha(game.t) * (self.R[s][a] + game.gamma * V - Q[a])
        self.update_policy(s, a, id_, game)
        self.record_policy(s, id_, game)

        if game.is_symmetric and game.do_symmetry:
            self.do_symmetry(s, id_, game)

    def update_policy(self, s, a, id_, game):
        Q = self.Q[id_][s]
        self.pi[id_][s] = (Q == np.max(Q)).astype(np.double)

    def do_symmetry(self, s, id_, game):
        s_ = game.symmetric_state(s)
        for a in range(game.numactions(id_)):
            self.pi[1 - id_][s_][game.symmetric_action(a)] = self.pi[id_][s][a]
            self.Q[1 - id_][s_][game.symmetric_action(a)] = self.Q[id_][s][a]


class PHCAgent(QAgent):
    def __init__(self, id_, game, delta=0.02):
        super().__init__(id_, game)
        self.name = 'phc'
        self.delta = delta

    def update_policy(self, s, a, id_, game):
        delta = self.delta * self.alpha(game.t)
        if a == np.argmax(self.Q[id_][s]):
            self.pi[id_][s][a] += delta
        else:
            self.pi[id_][s][a] -= delta / (game.numactions(id_) - 1)
        StationaryAgent.normalize(self.pi[id_][s])


class WoLFAgent(PHCAgent):
    def __init__(self, id_, game, delta1=0.01, delta2=0.04):
        super().__init__(id_, game)
        self.name = 'wolf'
        self.delta1 = delta1
        self.delta2 = delta2
        self.pi_ = defaultdict(partial(np.random.dirichlet, [1.0] * game.numactions(id_)))
        self.count_pi = defaultdict(int)

    def done(self, id_, game):
        self.pi_.clear()
        self.count_pi.clear()
        super().done(id_, game)

    def update_policy(self, s, a, id_, game):
        self.count_pi[s] += 1
        self.pi_[s] += (self.pi[id_][s] - self.pi_[s]) / self.count_pi[s]
        self.delta = self.delta1 \
            if np.dot(self.pi[id_][s], self.Q[id_][s]) \
               > np.dot(self.pi_[s], self.Q[id_][s]) \
            else self.delta2
        super().update_policy(s, a, id_, game)


class MinimaxQAgent(BaseQAgent):
    def __init__(self, id_, game):
        super().__init__('minimax', id_, game)
        self.solvers = []
        self.Q = np.array(
            [defaultdict(partial(np.random.rand, game.numactions(0), game.numactions(1))),
             defaultdict(partial(np.random.rand, game.numactions(1), game.numactions(0)))])

    def done(self, id_, game):
        self.solvers.clear()
        super().done(id_, game)

    def val(self, s, id_, game):
        Q = self.Q[id_][s]
        pi = self.pi[id_][s]
        return min(np.dot(pi, Q[:, o]) for o in range(game.numactions(1 - id_)))

    def update(self, s, a, o, r, sp, id_, game):
        Q = self.Q[id_][s]
        V = self.val(sp, id_, game)
        Q[a, o] += self.alpha(game.t) * (r + game.gamma * V - Q[a, o])
        self.update_policy(s, a, id_, game)
        self.record_policy(s, id_, game)

        if game.is_symmetric and game.do_symmetry:
            self.do_symmetry(s, id_, game)

    def update_policy(self, s, a, id_, game):
        self.initialize_solvers()
        for solver, lib in self.solvers:
            try:
                self.pi[id_][s] = MinimaxQAgent.lp_solve(
                    self.Q[id_][s], solver, lib, id_, game)
                StationaryAgent.normalize(self.pi[id_][s])
            except Exception as e:
                print('optimization using {} failed: {}'.format(solver, e))
                continue
            else: break

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
                try: self.solvers.append((lib, importlib.import_module(lib)))
                except: pass

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


class MetaControlAgent(Agent):
    def __init__(self, id_, game):
        super().__init__('metacontrol')
        self.agents = [QAgent(id_, game), MinimaxQAgent(id_, game)]
        self.r = np.zeros(len(self.agents))
        self.n = np.zeros(len(self.agents))

        self.controller = None
        self.step_within_episode = 0
        self.cumulative_r = 0

    def act(self, s, exploration, id_, game):
        if self.controller is None:
            self.controller = np.argmax([self.ucb(i, exploration) for i in range(len(self.agents))])
        return self.agents[self.controller].act(s, exploration, id_, game)

    def done(self, id_, game):
        for agent in self.agents:
            agent.done(id_, game)

    def ucb(self, i, exploration):
        if exploration:
            if self.n[i] == 0:
                return float('inf')
            return self.r[i] + math.sqrt(2.0 * math.log(np.sum(self.n)) / self.n[i])
        return self.r[i]

    def update(self, s, a, o, r, sp, id_, game):
        for agent in self.agents:
            agent.update(s, a, o, r, sp, id_, game)

        self.cumulative_r += math.pow(game.gamma, self.step_within_episode) * r
        self.step_within_episode += 1

        if game.new_episode:
            self.n[self.controller] += 1
            self.r[self.controller] += (self.cumulative_r - self.r[self.controller]) / self.n[self.controller]

            self.controller = None
            self.step_within_episode = 0
            self.cumulative_r = 0

            print('id: {}, n: {} ({}%), r: {}'.format(
                id_, self.n, 100.0 * self.n / np.sum(self.n), self.r))


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
