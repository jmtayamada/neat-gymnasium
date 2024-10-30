'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import argparse
import pickle
import time

import gymnasium as gym
from gymnasium import wrappers
import numpy as np

from neat.nn import FeedForwardNetwork

import importlib


def _gym_make(envname, wrappers: list[str] = []) -> gym.Env:

    env = None

    try:
        env = gym.make(envname)
        for wrapper in wrappers:
            wordList = wrapper.split(".")
            wrapperName = wordList.pop()
            
            baseModule = ""
            for word in wordList:
                if baseModule != "":
                    baseModule += "." + word
                else:
                    baseModule += word
            
            module = importlib.import_module(baseModule)
            wrapper_class = getattr(module, wrapperName)
            env = wrapper_class(env)

    except Exception as e:
        print('Unable to make environment %s: %s' %
              (envname, e))
        exit(1)

    return env


def _is_discrete(env):
    return 'Discrete' in str(type(env.action_space))


# Public functions ===================================================


def read_file(allow_record=False, allow_seed=False):
    '''
    Reads a genome/config file based on command-line argument
    @param allow_record set to enable --record option
    @param allow_seed set to enable --seed option
    @return net, env_name, recording flag, seed, no-display flag, CSV file name
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename', metavar='FILENAME', help='.dat input file')

    parser.add_argument('--nodisplay', dest='nodisplay', action='store_true',
                        help='Suppress display')
    if allow_record:
        parser.add_argument('--record', default=None,
                            help='If specified, sets the recording dir')

    if allow_seed:
        parser.add_argument('--seed', type=int, default=None,
                            help='Seed for random number generator')

    parser.add_argument('--save', dest='csvfilename',
                        help='Save trajectory in CSV file')

    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = pickle.load(open(args.filename, 'rb'))

    # Return genome, config, and optional save flag
    return (net,
            env_name,
            args.record if allow_record else None,
            args.seed if allow_seed else None,
            args.nodisplay,
            args.csvfilename,)


def eval_net(
        net: FeedForwardNetwork,
        env: gym.Env,
        render=False,
        report=False,
        record_dir=None,
        activations=1,
        seed=None,
        max_episode_steps=None,
        csvfilename=None,
        **kwargs):
    '''
    Evaluates a network
    @param net the network
    @param env the Gym environment
    @param render set to True for rendering
    @param record_dir set to directory name for recording video
    @param activations number of times to repeat
    @param seed seed for random number generator
    @param csvfilename name of CSV file for saving trajectory
    @return total reward
    '''
    
    for key, value in kwargs.items():
        if key == 'rendType':
            rendType = value

    if record_dir is not None:
        env = wrappers.Monitor(env, record_dir, force=True)

    # env.seed(seed)
    state, _ = env.reset()
    state = state.flatten()
    total_reward = 0
    steps = 0

    is_discrete = _is_discrete(env)

    csvfile = None

    if csvfilename is not None:

        csvfile = open(csvfilename, 'w')

    while max_episode_steps is None or steps < max_episode_steps:

        # Support recurrent nets
        for k in range(activations):
            action = net.activate(state)

        # Support both discrete and continuous actions
        action = (np.argmax(action)
                  if is_discrete else action * env.action_space.high)

        state, reward, terminated, truncated, _ = env.step(action)
        state = state.flatten()

        if csvfile is not None:

            if is_discrete:
                csvfile.write('%d,' % action)

            else:
                fmt = ('%f,' * len(action))
                csvfile.write(fmt % tuple(action))

            fmt = ('%f,' * len(state))[:-1] + '\n'
            csvfile.write(fmt % tuple(state))

        if render or (record_dir is not None):
            env.render()
            time.sleep(.02)

        total_reward += reward

        if terminated or truncated:
            break

        steps += 1

    if csvfile is not None:
        csvfile.close()

    env.close()

    if report:
        print('Got reward %+6.6f in %d steps' % (total_reward, steps))

    return total_reward, steps
