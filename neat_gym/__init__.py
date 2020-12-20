'''
Common code for using NEAT with OpenAI Gym environments

Copyright (C) 2020 Simon D. Levy

MIT License
'''

import multiprocessing as mp
import os
import argparse
import random
import pickle
import time
import warnings
from configparser import ConfigParser

import gym
from gym import wrappers
import neat
import numpy as np

from neat.config import ConfigParameter, UnknownConfigItemError

from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate

def _is_discrete(env):
    return 'Discrete' in str(type(env.action_space))

class _Config(object):
    #Adapted from https://github.com/CodeReclaimers/neat-python/blob/master/neat/config.py

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False)]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename, layout_dict):

        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn("Using default {!r} for '{!s}'".format(p.default, p.name),
                                  DeprecationWarning)
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" +
                                             "\n\t".join(unknown_list))
            raise UnknownConfigItemError(
                "Unknown (section 'NEAT') configuration item {!s}".format(unknown_list[0]))

        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))

        # Add layout (input/output) info
        for key in layout_dict:
            genome_dict[key] = layout_dict[key]

        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

    @staticmethod
    def eval_genome(genome, config, net, activations):

        fitness = 0

        for _ in range(config.reps):

            fitness += eval_net(net, config.env, activations=activations, seed=config.seed)

        return fitness / config.reps

class _GymConfig(_Config):

    def __init__(self, args, layout_dict, suffix=''):

        filename = args.cfgdir + '/' + args.env + suffix + '.cfg'

        if not os.path.isfile(filename):
            print('Unable to open config file ' + filename)
            exit(1)

        _Config.__init__(self, neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                filename, layout_dict)

        self.env_name = args.env
        self.env = gym.make(args.env)

        self.reps = args.reps
        self.seed = args.seed

        namescfg = _GymConfig.load(args, suffix)

        try:
            names =  namescfg['Names']
            self.node_names = {}
            for idx,name in enumerate(eval(names['input'])):
                self.node_names[-idx-1] = name
            for idx,name in enumerate(eval(names['output'])):
                self.node_names[idx] = name
        except:
            self.node_names = {}

    def save_genome(self, genome):

        name = self._make_name(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, self)
        pickle.dump((net, self.env_name), open('models/%s.dat' % name, 'wb'))
        _GymConfig._draw_net(net, 'visuals/%s'%name, self.node_names)

    def _make_name(self, genome, suffix=''):

        return '%s%s%+010.3f' % (self.env_name, suffix, genome.fitness)

    @staticmethod
    def eval_genome(genome, config):

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return _Config.eval_genome(genome, config, net, 1)

    @staticmethod
    def load(args, suffix):

        filename = args.cfgdir + '/' + args.env + suffix + '.cfg'
        if not os.path.isfile(filename):
            print('Cannot open config file ' + filename)
            exit(1)

        parser = ConfigParser()
        parser.read(filename)
        return parser

    @staticmethod 
    def _draw_net(net, filename, node_names):

        # Create PDF
        draw_net(net, filename=filename, node_names=node_names) 

        # Delete text
        os.remove(filename) 

    @staticmethod
    def make_config(args):

        # Get input/output layout from environment
        env = gym.make(args.env)
        num_inputs  = env.observation_space.shape[0]
        num_outputs = env.action_space.n if _is_discrete(env) else env.action_space.shape[0]

        # Load rest of config from file
        config = _GymConfig(args, {'num_inputs':num_inputs, 'num_outputs':num_outputs})
        evalfun = _GymConfig.eval_genome
     
        return config, evalfun

class _GymHyperConfig(_GymConfig):

    def __init__(self, args, substrate, actfun, suffix='-hyper'):

        _GymConfig.__init__(self, args, {'num_inputs':5, 'num_outputs':1}, suffix)

        self.substrate = substrate
        self.actfun = actfun

        # Output of CPPN is recurrent, so negate indices
        self.node_names = {j:self.node_names[k] for j,k in enumerate(self.node_names)} 

        # CPPN itself always has the same input and output nodes XXX are these correct?
        self.cppn_node_names = {-1:'x1', -2:'y1', -3:'x2', -4:'y2', -5:'bias', 0:'weight'}

    def save_genome(self, genome):

        cppn, net = _GymHyperConfig._make_nets(genome, self)
        self._save_nets(genome, cppn, net)

    def _save_nets(self, genome, cppn, net, suffix='-hyper'):
        pickle.dump((net, self.env_name), open('models/%s.dat' % self._make_name(genome, suffix=suffix), 'wb'))
        _GymConfig._draw_net(cppn, 'visuals/%s' % self._make_name(genome, suffix='-cppn'), self.cppn_node_names)
        _GymConfig._draw_net(net, 'visuals/%s' % self._make_name(genome, suffix=suffix), self.node_names)

    @staticmethod
    def eval_genome(genome, config):

        cppn, net = _GymHyperConfig._make_nets(genome, config)
        activations = len(config.substrate.hidden_coordinates) + 2
        return _Config.eval_genome(genome, config, net, activations)

    @staticmethod
    def _make_nets(genome, config):

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        return cppn, create_phenotype_network(cppn, config.substrate, config.actfun)

    @staticmethod
    def make_config(args):
        
        cfg = _GymConfig.load(args, '-hyper')
        subs =  cfg['Substrate']
        actfun = subs['function']
        inp = eval(subs['input'])
        hid = eval(subs['hidden'])
        out = eval(subs['output'])
        substrate = Substrate(inp, out, hid)

        # Load rest of config from file
        config = _GymHyperConfig(args, substrate, actfun)

        evalfun = _GymHyperConfig.eval_genome

        return config, evalfun
     
class _GymEsHyperConfig(_GymHyperConfig):

    def __init__(self, args, substrate, actfun, params):

        self.params = {
                'initial_depth'     : int(params['initial_depth']),
                'max_depth'         : int(params['max_depth']),
                'variance_threshold': float(params['variance_threshold']),  
                'band_threshold'    : float(params['band_threshold']),  
                'iteration_level'   : int(params['iteration_level']),  
                'division_threshold': float(params['division_threshold']),  
                'max_weight'        : float(params['max_weight']),
                'activation'        : params['activation']  
                }

        _GymHyperConfig.__init__(self, args, substrate, actfun, suffix='-eshyper')

    def save_genome(self, genome):

        cppn, _, net = _GymEsHyperConfig._make_nets(genome, self)
        _GymEsHyperConfig._save_nets(self, genome, cppn, net, suffix='-eshyper')

    @staticmethod
    def eval_genome(genome, config):

        _, esnet, net = _GymEsHyperConfig._make_nets(genome, config)
        return _Config.eval_genome(genome, config, net, esnet.activations)

    @staticmethod
    def _make_nets(genome, config):

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        esnet = ESNetwork(config.substrate, cppn, config.params)
        net = esnet.create_phenotype_network()
        return cppn, esnet, net

    @staticmethod
    def make_config(args):

        # Load config from file
        cfg = _GymConfig.load(args, '-eshyper')
        subs =  cfg['Substrate']
        actfun = subs['function']
        inp = eval(subs['input'])
        out = eval(subs['output'])

        # Get substrate from -hyper.cfg file named by Gym environment
        substrate = Substrate(inp, out)

        # Load rest of config from file
        config = _GymEsHyperConfig(args, substrate, actfun, cfg['ES'])

        evalfun = _GymEsHyperConfig.eval_genome

        return config, evalfun

class _SaveReporter(neat.reporting.BaseReporter):

    def __init__(self, env_name, checkpoint):

        neat.reporting.BaseReporter.__init__(self)

        self.best = None
        self.env_name = env_name
        self.checkpoint = checkpoint

    def post_evaluate(self, config, population, species, best_genome):

        if self.checkpoint and (self.best is None or best_genome.fitness > self.best):
            self.best = best_genome.fitness
            print('############# Saving new best %f ##############' % self.best)
            config.save_genome(best_genome)

def _evolve(configfun):

    # Parse command-line arguments

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', default='CartPole-v1', help='Environment id')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Save at each new best')
    parser.add_argument('--cfgdir', required=False, default='./config', help='Directory for config files')
    parser.add_argument('--ngen', type=int, required=False, help='Number of generations to run')
    parser.add_argument('--reps', type=int, default=10, required=False, help='Number of repetitions per genome')
    parser.add_argument('--seed', type=int, required=False, help='Seed for random number generator')
    args = parser.parse_args()

    # Set random seed (including None)
    random.seed(args.seed)

    # Make directories for saving results
    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    # Get configuration and genome evaluation function for a particular algorithm
    config, evalfun = configfun(args) 

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(show_species_detail=False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Add a reporter (which can also checkpoint the best)
    p.add_reporter(_SaveReporter(args.env, args.checkpoint))

    # Create a parallel fitness evaluator
    pe = neat.ParallelEvaluator(mp.cpu_count(), evalfun)

    # Run for number of generations specified in config file
    winner = p.run(pe.evaluate) if args.ngen is None else p.run(pe.evaluate, args.ngen) 

    # Save winner
    config.save_genome(winner)


def read_file(allow_record=False):
    '''
    Reads a genome/config file based on command-line argument
    @return genome,config tuple
    '''

    # Parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', metavar='FILENAME', help='.dat input file')
    parser.add_argument('--nodisplay', dest='nodisplay', action='store_true', help='Suppress display')
    if allow_record:
        parser.add_argument('--record', default=None, help='If specified, sets the recording dir')
    args = parser.parse_args()

    # Load net and environment name from pickled file
    net, env_name = pickle.load(open(args.filename, 'rb'))

    # Return genome, config, and optional save flag
    return net, env_name, args.record if allow_record else None, args.nodisplay

def eval_net(net, env, render=False, record_dir=None, activations=1, seed=None):
    '''
    Evaluates an evolved network
    @param net the network
    @param env the Gym environment
    @param render set to True for rendering
    @param record_dir set to directory name for recording video
    @param actviations number of times to repeat
    @param seed seed for random number generator
    @return total reward
    '''

    if record_dir is not None:
        env = wrappers.Monitor(env, record_dir, force=True)

    env.seed(seed)
    state = env.reset()
    total_reward = 0
    steps = 0

    is_discrete = _is_discrete(env)

    while True:

        # Support recurrent nets
        for k in range(activations): 
            action = net.activate(state)

        # Support both discrete and continuous actions
        action = np.argmax(action) if is_discrete else action * env.action_space.high

        state, reward, done, _ = env.step(action)
        if render:
            env.render('rgb_array')
            time.sleep(.02)
        total_reward += reward
        if done:
            break
        steps += 1

    env.close()

    return total_reward


