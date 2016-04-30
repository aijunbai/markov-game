# coding=utf-8

import random

import numpy as np

import markovgame
import utils

__author__ = 'Aijun Bai'


class State(object):
    def __init__(self):
        self.ball = 0  # the player holding the ball
        self.positions = {0: np.array([0, 0]), 1: np.array([0, 0])}

    def clone(self):
        cloned = State()
        cloned.ball = self.ball
        cloned.positions[0] = np.copy(self.positions[0])
        cloned.positions[1] = np.copy(self.positions[1])
        return cloned

    def __str__(self):
        return 'b:{} p:{}'.format(self.ball, self.positions)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not (self == other)


class Simulator(markovgame.Simulator):
    directions = {0: np.array([0, 1]),  # N
                  1: np.array([1, 0]),  # E
                  2: np.array([0, -1]),  # S
                  3: np.array([-1, 0])  # W
                  }

    def __init__(self):
        super().__init__(numactions=5)
        self.length = 5
        self.width = 4

    def random_position(self):
        return np.array([random.randint(0, self.length - 1), random.randint(0, self.width - 1)])

    def initial_state(self):
        state = State()
        state.ball = random.randint(0, 1)

        while np.array_equal(state.positions[0], state.positions[1]):
            state.positions[0] = self.random_position()
            state.positions[1] = self.random_position()

        self.assertion(state)
        return state

    def validate(self, pos):
        pos[0] = utils.minmax(0, pos[0], self.length - 1)
        pos[1] = utils.minmax(0, pos[1], self.width - 1)

    def assertion(self, state):
        assert 0 <= state.positions[0][0] <= self.length - 1
        assert 0 <= state.positions[0][1] <= self.width - 1
        assert 0 <= state.positions[1][0] <= self.length - 1
        assert 0 <= state.positions[1][1] <= self.width - 1
        assert not np.array_equal(state.positions[0], state.positions[1])
        assert state.ball == 0 or state.ball == 1

    def goal(self, state, actions):
        if 0 <= actions[state.ball] <= 3:
            pos = state.positions[state.ball] + Simulator.directions[actions[state.ball]]
            if state.ball == 0 and pos[0] < 0 and (self.width - 1) / 2 <= pos[1] <= (self.width + 1) / 2:
                return 0
            elif state.ball == 1 and pos[0] > self.length - 1 and (self.width - 1) / 2 <= pos[1] <= (
                self.width + 1) / 2:
                return 1
        return None

    def step(self, state, actions, verbose=False):
        rewards = {no: 0 for no in actions}
        next_state = state.clone()

        for i in actions:
            if 0 <= actions[i] <= 3:
                next_state.positions[i] += Simulator.directions[actions[i]]
                self.validate(next_state.positions[i])

        if np.array_equal(next_state.positions[0], next_state.positions[1]):
            next_state = state.clone()  # the move does not take place
            if actions[0] == 4:
                next_state.ball = 0
            elif actions[1] == 4:
                next_state.ball = 1

        goal = self.goal(state, actions)
        if goal is not None:
            rewards[goal] = 1
            rewards[1 - goal] = -1
            next_state = self.initial_state()

        self.assertion(next_state)
        if verbose:
            self.draw(state)
            print('actions: {}'.format(actions))
            print('rewards: {}'.format(rewards))

        return next_state, rewards

    def draw(self, state):
        for r in range(-1, self.width + 1):
            for c in range(-1, self.length + 1):
                if 0 <= r <= self.width - 1 and 0 <= c <= self.length - 1:
                    if np.array_equal(state.positions[0], np.array([c, self.width - 1 - r])):
                        print('A' if state.ball == 0 else 'a', end='')
                    elif np.array_equal(state.positions[1], np.array([c, self.width - 1 - r])):
                        print('B' if state.ball == 1 else 'b', end='')
                    else:
                        print('.', end='')
                else:
                    print('#', end='')
            print('\n', end='')
        print(state)


class LittmanSoccer(markovgame.MarkovGame):
    def __init__(self, H):
        self.simulator = Simulator()
        super().__init__('littmansoccer', self.simulator, 0.9, H)

    def numactions(self, no):
        return self.simulator.numactions
