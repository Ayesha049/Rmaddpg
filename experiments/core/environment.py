"""
Environment and model utilities for MADDPG experiments.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from gym.spaces import Box, Discrete
from maddpg.trainer.maddpg import MADDPGAgentTrainer


def get_total_action_dim(env):
    """
    Return total joint action dimension across all agents.
    Handles both Box and Discrete action spaces.
    """
    total = 0
    for i in range(env.n):
        space = env.action_space[i]
        if isinstance(space, Box):
            total += int(np.prod(space.shape))
        elif isinstance(space, Discrete):
            total += space.n
        else:
            raise NotImplementedError(
                "Unsupported action space type: {}".format(type(space))
            )
    return total


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    """
    Multi-layer perceptron model for MADDPG agents.
    Takes as input an observation and returns values of all actions.
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    """
    Create a multi-agent environment for the given scenario.
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    """
    Create MADDPG agent trainers for all agents in the environment.
    """
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer

    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))

    return trainers