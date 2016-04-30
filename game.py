# coding=utf-8


from abc import ABCMeta, abstractmethod

__author__ = 'Aijun Bai'


class Game(object, metaclass=ABCMeta):
    def __init__(self, name, gamma, H):
        self.name = name
        self.gamma = gamma
        self.H = int(H)
        self.players = set()
        self.state = None

    def add_player(self, player):
        self.players.add(player)

    @abstractmethod
    def numactions(self, no):
        pass

    def run(self):
        assert len(self.players) == 2

        for i in range(self.H):
            print('step: {}'.format(i))
            actions = {player.no: player.act(self.state, True) for player in self.players}
            next_state, rewards = self.simulate(actions, verbose=True)

            for player in self.players:
                j = player.no
                player.update(self.state, actions[j], actions[1 - j], rewards[j], next_state)  # TODO: add states

            self.state = next_state

    @abstractmethod
    def simulate(self, actions, verbose=False):  # state, actions -> state, reward
        pass
