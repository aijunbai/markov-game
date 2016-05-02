# coding=utf-8

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import random
from numba import jit

import numpy as np
import markovgame
import utils

__author__ = 'Aijun Bai'

@jit(nopython=True)
def jit_one_step(positions, directions, actions, ball):
    return positions[ball] + directions[actions[ball], :]

@jit(nopython=True)
def jit_joint_one_step(positions, directions, actions, l, w):
    positions[0] += directions[actions[0], :]
    positions[1] += directions[actions[1], :]
    jit_validate(positions[0], l, w)
    jit_validate(positions[1], l, w)

@jit(nopython=True)
def jit_validate(position, l, w):
    position[0] = min(max(0, position[0]), l - 1)
    position[1] = min(max(0, position[1]), w - 1)

@jit(nopython=True)
def jit_position_equal(positions):
    return positions[0][0] == positions[1][0] and positions[0][1] == positions[1][1]

@jit(nopython=True)
def jit_positions_equal(s1, s2):
    return s1[0][0] == s2[0][0] and s1[0][1] == s2[0][1] \
           and s1[1][0] == s2[1][0] and s1[1][1] == s2[1][1]

@jit(nopython=True)
def jit_random_position(l, w):
    return np.array([np.random.randint(0, l), np.random.randint(0, w)], dtype=np.int)


class State(object):
    def __init__(self):
        self.ball = 0  # the player holding the ball
        self.positions = np.zeros((2, 2), dtype=np.int)

    def clone(self):
        cloned = State()
        cloned.ball = self.ball
        cloned.positions = np.copy(self.positions)
        return cloned

    def __str__(self):
        return 'b:{} p:{}'.format(self.ball, self.positions)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.ball, self.positions.tostring()))

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
               and self.ball == other.ball \
               and jit_positions_equal(self.positions, other.positions)

    def __ne__(self, other):
        return not (self == other)


class Simulator(markovgame.Simulator):
    directions = np.array([[0, 1],
                           [1, 0],
                           [0, -1],
                           [-1, 0],
                           [0, 0]], dtype=np.int)

    def __init__(self, game):
        super().__init__(game)
        self.length = 5
        self.width = 4
        self.random_threshold = 0.1
        self.episodes = 1
        self.wins = np.ones(2, dtype=np.int)

    def numactions(self, no):
        return 5

    def report(self):
        print('step: {}'.format(self.game.step))
        print('episodes: {}'.format(self.episodes))
        for i in range(2):
            print('{}: win {} ({}%)'.format(
                i, self.wins[i], self.wins[i] / self.episodes * 100))

    def random_position(self):
        return jit_random_position(self.length, self.width)

    def initial_state(self, random_positions=False):
        state = State()
        state.ball = random.randint(0, 1)

        if random_positions:
            while jit_position_equal(state.positions):
                state.positions[0] = self.random_position()
                state.positions[1] = self.random_position()
        else:
            state.positions[0] = np.array([3, 2], dtype=np.int)
            state.positions[1] = np.array([1, 1], dtype=np.int)

        self.assertion(state)
        return state

    def assertion(self, state):
        assert 0 <= state.positions[0][0] <= self.length - 1
        assert 0 <= state.positions[0][1] <= self.width - 1
        assert 0 <= state.positions[1][0] <= self.length - 1
        assert 0 <= state.positions[1][1] <= self.width - 1
        assert not jit_position_equal(state.positions)
        assert state.ball == 0 or state.ball == 1

    def goal(self, state, actions):
        pos = jit_one_step(state.positions, Simulator.directions, actions, state.ball)

        if (self.width - 1) / 2 <= pos[1] <= (self.width + 1) / 2:
            if state.ball == 0 and pos[0] < 0:
                return 0
            elif state.ball == 1 and pos[0] > self.length - 1:
                return 1

        return None

    def step(self, state, actions, verbose=False):
        self.assertion(state)

        rewards = np.zeros(2)
        goal = self.goal(state, actions)

        if goal is not None:
            rewards[goal] = 1
            rewards[1 - goal] = -1
            next_state = self.initial_state()  # new episode
            self.episodes += 1
            self.wins[goal] += 1
            self.report()
        else:
            next_state = state.clone()

            for i in range(2):
                if random.random() < self.random_threshold:
                    actions[i] = random.randint(0, self.numactions(i) - 1)

            jit_joint_one_step(
                next_state.positions,
                Simulator.directions,
                actions, self.length, self.width)

            if jit_position_equal(next_state.positions):
                next_state = state.clone()  # the move does not take place

                if actions[0] == 4:
                    next_state.ball = 0
                elif actions[1] == 4:
                    next_state.ball = 1

        if verbose:
            self.draw(state)
            print('actions: {}'.format(actions))
            print('rewards: {}'.format(rewards))

        return next_state, rewards

    def draw(self, state):
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
        print(state)


class LittmanSoccer(markovgame.MarkovGame):
    def __init__(self, H):
        self.simulator = Simulator(self)
        super().__init__('littmansoccer', self.simulator, 0.9, H)

    def numactions(self, no):
        return self.simulator.numactions(no)
