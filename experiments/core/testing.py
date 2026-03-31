"""
Testing and evaluation functions for MADDPG experiments.
"""
import numpy as np
import tensorflow as tf
import time
import csv
import os

import maddpg.common.tf_util as U


def setup_environment_and_trainers(arglist):
    """
    Common setup for environment and agent trainers.

    Returns:
        env: Multi-agent environment
        trainers: List of agent trainers
        obs_shape_n: List of observation shapes
    """
    from .environment import make_env, get_trainers

    # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)

    # Create agent trainers
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    num_adversaries = min(env.n, arglist.num_adversaries)
    trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

    return env, trainers, obs_shape_n


def initialize_and_load_model(arglist, trainers):
    """
    Initialize TensorFlow graph and load trained model.
    """
    print('Testing using good policy {} and adv policy {}'.format(
        arglist.good_policy, arglist.adv_policy))

    # Initialize TF graph
    U.initialize()

    # Load trained model
    arglist.load_dir = arglist.save_dir
    print('Loading trained model from {}'.format(arglist.load_dir))
    U.load_state(arglist.load_dir, exp_name=arglist.exp_name)


def setup_testing_environment(env):
    """
    Setup environment variables for testing with disruptions.
    """
    env.llm_disturb_iteration = 0
    env.previous_reward = 0


def run_episode_loop(env, trainers, arglist, episode_idx, apply_disruptions_func=None):
    """
    Run a single episode with optional disruption functions.

    Args:
        apply_disruptions_func: Function that takes (obs_n, action_n, env, arglist)
                               and returns (obs_n, action_n) after applying disruptions

    Returns:
        episode_reward: Array of rewards for each agent in the episode
    """
    obs_n = env.reset()
    episode_reward = np.zeros(env.n)

    for step in range(arglist.max_episode_len):
        # Apply disruptions if function provided
        if apply_disruptions_func:
            obs_n, action_n = apply_disruptions_func(obs_n, None, env, arglist, step)
        else:
            # Get actions from trained policies
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

        # Environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        # Track reward
        episode_reward += rew_n
        obs_n = new_obs_n

        # Render if needed
        if arglist.display:
            env.render()
            time.sleep(0.05)

        if all(done_n):
            break

    return episode_reward


def calculate_and_report_results(all_rewards, n_episodes):
    """
    Calculate mean rewards and print results.

    Returns:
        average_total_reward: Mean total reward across all episodes
    """
    mean_rewards = np.mean(all_rewards, axis=0)
    print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
    average_total_reward = np.mean(np.sum(all_rewards, axis=1))
    print("Average total reward: {}".format(average_total_reward))
    return average_total_reward


def apply_observation_only_disruptions(obs_n, action_n, env, arglist, step, trainers):
    """
    Apply observation disruptions only.
    """
    from .noise import apply_observation_disruption

    # Apply observation disruption
    disrupted_obs_n = []
    for i, obs in enumerate(obs_n):
        reward = 0 if step == 0 else env.previous_reward  # Use appropriate reward
        disrupted_obs_n.append(apply_observation_disruption(obs, reward, env, arglist))

    obs_n = disrupted_obs_n
    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]

    return obs_n, action_n


def apply_observation_action_disruptions(obs_n, action_n, env, arglist, step, trainers):
    """
    Apply both observation and action disruptions.
    """
    from .noise import apply_observation_disruption, apply_action_disruption

    # Apply observation disruption before action selection
    obs_n_disrupted = [
        apply_observation_disruption(obs, 0, env, arglist)
        for obs in obs_n
    ]

    # Get actions from agents
    action_n = [
        agent.action(obs_dis)
        for agent, obs_dis in zip(trainers, obs_n_disrupted)
    ]

    # Apply action disruption
    action_n_disrupted = [
        apply_action_disruption(action, 0, env, arglist)
        for action in action_n
    ]

    return obs_n, action_n_disrupted


def apply_action_with_diffusion_disruptions(obs_n, action_n, env, arglist, step, trainers, diffusion_model, t_start=40):
    """
    Apply action disruptions with optional diffusion denoising.
    """
    from .noise import apply_action_disruption
    from .diffusion import diffusion_denoise_action, concat_actions, split_actions

    # Get actions from agents
    action_n = [
        agent.action(obs)
        for agent, obs in zip(trainers, obs_n)
    ]

    # Number of agents and action dimensions
    n_agents = len(action_n)
    action_dim_per_agent = [len(a) for a in action_n]

    # Apply adversarial noise
    action_n_noisy = [
        apply_action_disruption(action, 0, env, arglist)
        for action in action_n
    ]

    action_n_clean = action_n_noisy

    if diffusion_model:
        # Diffusion denoising
        action_vec_noisy = concat_actions(action_n_noisy)
        state_vec = np.concatenate(obs_n, axis=0)

        action_vec_clean = diffusion_denoise_action(
            action_vec_noisy,
            state_vec,
            t_start=t_start
        )

        action_n_clean = split_actions(action_vec_clean, n_agents, action_dim_per_agent)

    return obs_n, action_n_clean


def testWithoutP(arglist):
    """
    Test MADDPG agents without any perturbations.
    """
    # Reset TensorFlow graph to avoid variable conflicts
    tf.reset_default_graph()

    # Setup
    env, trainers, obs_shape_n = setup_environment_and_trainers(arglist)

    with U.single_threaded_session():
        initialize_and_load_model(arglist, trainers)

        # Testing parameters
        n_episodes = arglist.num_test_episodes
        all_rewards = []

        print('Starting testing...')

        for ep in range(n_episodes):
            episode_reward = run_episode_loop(env, trainers, arglist, ep)
            all_rewards.append(episode_reward)

    return calculate_and_report_results(all_rewards, n_episodes)


def testRobustnessOP(arglist):
    """
    Test robustness with observation perturbations only.
    """
    # Reset TensorFlow graph to avoid variable conflicts
    tf.reset_default_graph()

    # Setup
    env, trainers, obs_shape_n = setup_environment_and_trainers(arglist)
    setup_testing_environment(env)

    with U.single_threaded_session():
        initialize_and_load_model(arglist, trainers)

        # Testing parameters
        n_episodes = arglist.num_test_episodes
        all_rewards = []

        print('Starting testing with observation perturbations...')

        for ep in range(n_episodes):
            episode_reward = run_episode_loop(
                env, trainers, arglist, ep,
                apply_disruptions_func=lambda obs_n, action_n, env, arglist, step:
                    apply_observation_only_disruptions(obs_n, action_n, env, arglist, step, trainers)
            )
            all_rewards.append(episode_reward)

    return calculate_and_report_results(all_rewards, n_episodes)


def testRobustnessOA(arglist):
    """
    Test robustness with observation and action perturbations.
    """
    # Reset TensorFlow graph to avoid variable conflicts
    tf.reset_default_graph()

    # Setup
    env, trainers, obs_shape_n = setup_environment_and_trainers(arglist)
    setup_testing_environment(env)

    with U.single_threaded_session():
        initialize_and_load_model(arglist, trainers)

        # Testing parameters
        n_episodes = arglist.num_test_episodes
        all_rewards = []

        print('Starting testing with observation and action perturbations...')

        for ep in range(n_episodes):
            episode_reward = run_episode_loop(
                env, trainers, arglist, ep,
                apply_disruptions_func=lambda obs_n, action_n, env, arglist, step:
                    apply_observation_action_disruptions(obs_n, action_n, env, arglist, step, trainers)
            )
            all_rewards.append(episode_reward)

    return calculate_and_report_results(all_rewards, n_episodes)


def testRobustnessAP(arglist, deffusion=True, t_start=40):
    """
    Test robustness with action perturbations and optional diffusion denoising.
    """
    # Reset TensorFlow graph to avoid variable conflicts
    tf.reset_default_graph()

    from .diffusion import load_diffusion_model

    # Setup
    env, trainers, obs_shape_n = setup_environment_and_trainers(arglist)
    setup_testing_environment(env)

    with U.single_threaded_session():
        initialize_and_load_model(arglist, trainers)

        if deffusion:
            load_diffusion_model(arglist)

        # Testing parameters
        n_episodes = arglist.num_test_episodes
        all_rewards = []

        print('Starting testing with action perturbations{}...'.format(
            ' and diffusion denoising' if deffusion else ''))

        for ep in range(n_episodes):
            episode_reward = run_episode_loop(
                env, trainers, arglist, ep,
                apply_disruptions_func=lambda obs_n, action_n, env, arglist, step:
                    apply_action_with_diffusion_disruptions(obs_n, action_n, env, arglist, step, trainers, deffusion, t_start)
            )
            all_rewards.append(episode_reward)

    return calculate_and_report_results(all_rewards, n_episodes)