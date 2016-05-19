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
  -L, --left_name STR      specify name for the left agent [default: ]
  -R, --right_name STR     specify name for the right agent [default: ]
  -t, --trainall           train both left and right agents
  -n, --numplots N         plot a number of policies for states [default: 0]
  -m, --max_steps N        run the simulation for M steps [default: 10k]
  -a, --animation          enable animation mode
  -d, --do_symmetry        do symmetry for Q agent if possible
  -s, --seed SEED          use SEED as the random seed [default: 0]
  -v, --verbose            operate in verbose mode
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

import utils
import bimatrixgame
import littmansoccer
import signal
import agent
import humanfriendly
import sys
import os
import pickle
from docopt import docopt

__author__ = 'Aijun Bai'


def create_game(game_name, *args, **kwargs):
    if game_name == 'penaltyshoot':
        return bimatrixgame.PenaltyShoot(*args, **kwargs)
    elif game_name == 'rockpaperscissors':
        return bimatrixgame.RockPaperScissors(*args, **kwargs)
    elif game_name == 'rockpaperscissorsspocklizard':
        return bimatrixgame.RockPaperScissorsSpockLizard(*args, **kwargs)
    elif game_name == 'matchingpennies':
        return bimatrixgame.MatchingPennies(*args, **kwargs)
    elif game_name == 'inspection':
        return bimatrixgame.Inspection(*args, **kwargs)
    elif game_name.startswith('random'):
        rows, cols = [int(s.strip('random')) for s in game_name.split('x')]
        return bimatrixgame.RandomGame(rows, cols, *args, **kwargs)
    elif game_name == 'littmansoccer':
        return littmansoccer.LittmanSoccer(*args, **kwargs)
    else:
        print('no such game: {}'.format(game_name))
        return None


def create_agent(agent_type, *args, **kwargs):
    if agent_type.startswith('stationary'):
        pi = None
        if agent_type.endswith(']'):
            pi = agent_type.lstrip('stationary[').rstrip(']').split(',')
            pi = [float(p) for p in pi]
        return agent.StationaryAgent(*args, **kwargs, pi=pi)
    elif agent_type == 'random':
        return agent.RandomAgent(*args, **kwargs)
    elif agent_type == 'q':
        return agent.QAgent(*args, **kwargs)
    elif agent_type == 'phc':
        return agent.PHCAgent(*args, **kwargs)
    elif agent_type == 'wolf':
        return agent.WoLFAgent(*args, **kwargs)
    elif agent_type == 'minimaxq':
        return agent.MinimaxQAgent(*args, **kwargs)
    elif agent_type == 'littmansoccerhandcoded':
        return littmansoccer.HandCodedAgent(*args, **kwargs)
    elif agent_type.endswith('pickle'):
        return load_agent(agent_type)
    else:
        print('no such agent: {}'.format(agent_type))
        return None


def save_agent(a, pickle_name):
    print('saving agent to {}'.format(pickle_name))
    with open(pickle_name, 'wb') as f:
        pickle.dump(a, f, protocol=2)


def load_agent(file_name):
    print('loading agent from {}'.format(file_name))
    with open(file_name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1.1rc')

    max_steps = humanfriendly.parse_size(arguments['--max_steps'])
    modes = {0: arguments['--left'], 1: arguments['--right']}
    trainall = arguments['--trainall']
    agents = {0: arguments['<left>'], 1: arguments['<right>']}
    names = {0: arguments['--left_name'], 1: arguments['--right_name']}
    game = arguments['<game>']
    animation = arguments['--animation']
    do_symmetry = arguments['--do_symmetry']
    numplots = int(arguments['--numplots'])
    seed = int(arguments['--seed'])
    verbose = arguments['--verbose']

    utils.random_seed(seed)
    if trainall:
        modes[0] = modes[1] = True

    G = create_game(game, max_steps)

    def done(*args):
        if args and len(args):
            print('Caught signal {}'.format(args[0]))

        G.done()
        os.makedirs('data/', exist_ok=True)
        for j in range(2):
            if modes[j]:
                save_agent(
                    G.players[j],
                    'data/{}.pickle'.format(G.players[j].full_name(j, G)))
        exit()

    signal.signal(signal.SIGINT, done)

    for j in range(2):
        G.add_player(j, create_agent(agents[j], j, G))
        if len(names[j]):
            G.players[j].name = names[j]

    G.verbose = verbose
    G.animation = animation
    G.do_symmetry = do_symmetry
    G.numplots = numplots
    G.run(modes)

    done()

