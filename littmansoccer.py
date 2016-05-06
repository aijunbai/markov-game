# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import random

import numpy as np

import markovgame
import utils

__author__ = 'Aijun Bai'

class State(object):
    def __init__(self):
        self.ball = 0  # the player holding the ball
        self.positions = np.zeros((2, 2), dtype=np.int)

    def clone(self):
        cloned = State()
        cloned.ball = self.ball
        cloned.positions = np.copy(self.positions)
        return cloned

    @staticmethod
    def bound(coordinates, min_value, max_value):
        np.clip(coordinates, min_value, max_value, out=coordinates)

    def __str__(self):
        return '{} {} {} {} {}'.format(
            self.ball,
            self.positions[0, 0],
            self.positions[0, 1],
            self.positions[1, 0],
            self.positions[1, 1])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.ball, self.positions.tostring()))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.ball == other.ball \
               and np.array_equal(self.positions, other.positions)

    def __ne__(self, other):
        return not (self == other)


class Simulator(markovgame.Simulator):
    directions = np.array(
        [[0, 1],   # N
         [1, 0],   # E
         [0, -1],  # S
         [-1, 0],  # W
         [0, 0]],  # stand
        dtype=np.int)

    action_names = {
        0: 'N',
        1: 'E',
        2: 'S',
        3: 'W',
        4: 'stand'}

    def __init__(self, game):
        super().__init__(game)
        self.length = 5
        self.width = 4
        self.bounds = np.array(
            [[0, self.length - 1],   # x
             [0, self.width - 1]],   # y
            dtype=np.int)

        self.episodes = 2
        self.wins = np.ones(2, dtype=np.int)

    def numactions(self, no):
        return 5

    def goal(self, i):
        self.episodes += 1
        self.wins[i] += 1
        self.report()

    def report(self):
        print('goal @ step: {}'.format(self.game.step))
        print('episodes: {}'.format(self.episodes))
        for i in range(2):
            print('{}: win {} ({}%)'.format(
                i, self.wins[i], self.wins[i] / self.episodes * 100))

    def random_position(self):
        return np.array([
            np.random.randint(0, self.length),
            np.random.randint(0, self.width)],
            dtype=np.int)

    def initial_state(self, random_positions=False):
        state = State()
        state.ball = random.randint(0, 1)

        if random_positions:
            while np.array_equal(state.positions[0], state.positions[1]):
                state.positions[0] = self.random_position()
                state.positions[1] = self.random_position()
        else:
            state.positions[0] = np.array([3, 2], dtype=np.int)
            state.positions[1] = np.array([1, 1], dtype=np.int)

        self.assertion(state)
        return state

    def assertion(self, state):
        for d in range(2):
            assert (self.bounds[d, 0] <= state.positions[:, d]).all()
            assert (state.positions[:, d] <= self.bounds[d, 1]).all()
        assert not np.array_equal(state.positions[0], state.positions[1])
        assert state.ball == 0 or state.ball == 1

    def is_goal(self, position, i):
        mid = 0.5 * (self.bounds[1, 0] + self.bounds[1, 1])
        if mid - 1.0 <= position[1] <= mid + 1.0:
            return position[0] < self.bounds[0, 0] if i == 0 else position[0] > self.bounds[0, 1]
        return False

    def bound(self, position):
        for d in range(2):
            position[d] = utils.minmax(self.bounds[d, 0], position[d], self.bounds[d, 1])

    def step(self, state, actions):
        self.assertion(state)

        rewards = np.zeros(2)
        state_prime = state.clone()
        order = [0, 1] if random.random() < 0.5 else [1, 0]
        goal_player = None

        for i in order:
            assert 0 <= actions[i] < self.numactions(i)
            position = state_prime.positions[i] + Simulator.directions[actions[i], :]

            if state_prime.ball == i and self.is_goal(position, i):  # goal
                rewards[i] = 1
                rewards[1 - i] = -1
                state_prime = self.initial_state()  # new episode
                goal_player = i
                break
            else:
                self.bound(position)

                if np.array_equal(position, state_prime.positions[1 - i]):  # the move does not take place
                    state_prime.ball = 1 - state_prime.ball  # swithch ball possession
                else:
                    state_prime.positions[i] = position

        if self.game.verbose:
            self.draw('state', state)
            print('actions: {}'.format([Simulator.action_names[a] for a in actions]))
            print('order: {}'.format(order))
            print('rewards: {}'.format(rewards))
            print()

        if goal_player is not None:
            self.goal(goal_player)
            print()

        return state_prime, rewards

    def draw(self, title, state):
        print('{}: {}'.format(title, state))
        for r in range(-1, self.width + 1):
            for c in range(-1, self.length + 1):
                if 0 <= r <= self.width - 1 and 0 <= c <= self.length - 1:
                    x, y = c, self.width - 1 - r
                    if np.array_equal(state.positions[0], np.array([x, y])):
                        print('A' if state.ball == 0 else 'a', end='')
                    elif np.array_equal(state.positions[1], np.array([x, y])):
                        print('B' if state.ball == 1 else 'b', end='')
                    else:
                        print('.', end='')
                else:
                    print('#', end='')
            print('\n', end='')


class LittmanSoccer(markovgame.MarkovGame):
    def __init__(self, H):
        self.simulator = Simulator(self)
        super().__init__('littmansoccer', self.simulator, 0.9, H)

    def numactions(self, no):
        return self.simulator.numactions(no)
