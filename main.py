# coding=utf-8

"""Usage:
  main.py [options] <game> <left> <right>
  main.py -h | --help | --version

Arguments:
  <game>                   run <game> as the game
  <left>                   run <left> as the left agent
  <right>                  run <right> as the right agent

Options:
  -h --help                show this help message and exit
  --version                show version and exit
  -l, --left               train the left agent
  -r, --right              train the right agent
  -t, --trainall           train both left and right agents
  -H, --horizon H          run the simulation for N steps [default: 100000]
  -R, --runs R             run the experiment R times [default: 1]
  -a, --animation          run the experiment in animation mode
  -s, --seed SEED          use SEED as the random seed [default: 0]
  -v, --verbose            operate in verbose mode
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

import utils
import bimatrixgame
import littmansoccer
import agent
import humanfriendly
import pickle
from docopt import docopt

__author__ = 'Aijun Bai'


def create_game(name, *args, **kwargs):
    if name == 'penaltyshoot':
        return bimatrixgame.PenaltyShoot(*args, **kwargs)
    elif name == 'rockpaperscissors':
        return bimatrixgame.RockPaperScissors(*args, **kwargs)
    elif name == 'rockpaperscissorsspocklizard':
        return bimatrixgame.RockPaperScissorsSpockLizard(*args, **kwargs)
    elif name == 'matchingpennies':
        return bimatrixgame.MatchingPennies(*args, **kwargs)
    elif name == 'inspection':
        return bimatrixgame.Inspection(*args, **kwargs)
    elif name.find('random') != -1:
        rows, cols = [int(s.strip('random')) for s in name.split('x')]
        return bimatrixgame.RandomGame(rows, cols, *args, **kwargs)
    elif name == 'littmansoccer':
        return littmansoccer.LittmanSoccer(*args, **kwargs)
    else:
        print('no such game: {}'.format(name))
        return None


def create_agent(name, *args, **kwargs):
    if name == 'stationary':
        return agent.StationaryAgent(*args, **kwargs)
    elif name == 'random':
        return agent.RandomAgent(*args, **kwargs)
    elif name == 'q':
        return agent.QAgent(*args, **kwargs)
    elif name == 'minimaxq':
        return agent.MinimaxQAgent(*args, **kwargs)
    elif name == 'littmansoccerhandcoded':
        return littmansoccer.HandCodedAgent(*args, **kwargs)
    elif name.find('pickle') != -1:
        return load_agent(name)
    else:
        print('no such agent: {}'.format(name))
        return None


def save_agent(g, a):
    print('saving agent to {}'.format(a.pickle_name(g.name, g.H)))
    with open(a.pickle_name(g.name, g.H), 'wb') as f:
        pickle.dump(a, f, protocol=2)


def load_agent(file_name):
    print('loading agent from {}'.format(file_name))
    with open(file_name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1.1rc')

    R = humanfriendly.parse_size(arguments['--runs'])
    H = humanfriendly.parse_size(arguments['--horizon'])
    modes = {0: arguments['--left'], 1: arguments['--right']}
    trainall = arguments['--trainall']
    agents = {0: arguments['<left>'], 1: arguments['<right>']}
    game = arguments['<game>']
    animation = arguments['--animation']
    seed = int(arguments['--seed'])
    verbose = arguments['--verbose']

    utils.random_seed(seed)

    if trainall:
        modes[0] = modes[1] = True

    for i in range(R):
        g = create_game(game, H)

        for j in range(2):
            g.add_player(j, create_agent(agents[j], j, g))

        g.set_verbose(verbose)
        g.set_animation(animation)
        g.run(modes)

        for j in range(2):
            if modes[j]:
                save_agent(g, g.players[j])
