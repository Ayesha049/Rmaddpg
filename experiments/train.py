import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


import requests
import json
import csv
import os
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym.spaces import Box, Discrete


# ================= Diffusion globals =================
DIFFUSION_MODEL = None
DIFFUSION_CONSTS = {}
DIFFUSION_DEVICE = torch.device("cpu")


API_KEY = ""

def gpt_call(prompt):
    # url = "https://api.openai.com/v1/chat/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "Bearer {}".format(API_KEY)
    # }
    # data = {
    #     "model": "gpt-3.5-turbo",  # updated supported model
    #     "messages": [
    #         {"role": "system", "content": "You are an adversarial perturbation generator for robust RL. Output only the revised observation as a Python list."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     "temperature": 0.7,
    #     "max_tokens": 200
    # }

    # try:
    #     response = requests.post(url, headers=headers, data=json.dumps(data))
    #     result = response.json()

    #     if "error" in result:
    #         print("OpenAI API error:", result["error"])
    #         return None

    #     if "choices" not in result or len(result["choices"]) == 0:
    #         print("OpenAI API returned no choices:", result)
    #         return None

    #     # gpt-3.5-turbo returns content here:
    #     return result["choices"][0]["message"]["content"].strip()

    # except Exception as e:
    #     print("GPT call failed:", str(e))
    #     return None

    return None



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=200, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="predator-pray", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./model", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    parser.add_argument("--run-id", type=int, default=0, help="ID of the run for multiple seeds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # parser.add_argument("--mode", choices=["train", "test"], default="train", help="Run mode: 'train' to train agents, 'test' to run evaluation")
    parser.add_argument(
    "--mode",
    choices=["train", "test", "collect_diffusion", "train_diffusion"],
    default="train",
    help="Run mode:\n"
         "  'train'           : normal MADDPG training\n"
         "  'test'            : robustness tests\n"
         "  'collect_diffusion': roll out trained MADDPG and save trajectories\n"
         "  'train_diffusion' : train diffusion model on saved trajectories"
    )

    parser.add_argument("--num_test_runs", type=int, default=3, help="Number of test runs to perform when in test mode")



    # --- Robustness settings ---
    parser.add_argument("--noise-factor", type=str, default="state", choices=["none", "state", "reward"],
                        help="where to apply noise (state/reward/none)")
    parser.add_argument("--noise-type", type=str, default="gauss", choices=["gauss", "shift", "uniform"],
                        help="type of noise distribution")
    parser.add_argument("--noise-mu", type=float, default=0.0, help="mean for Gaussian noise")
    parser.add_argument("--noise-sigma", type=float, default=1, help="std for Gaussian noise")
    parser.add_argument("--act-noise", type=float, default=1, help="std for Gaussian noise")
    parser.add_argument("--noise-shift", type=float, default=0.9, help="shift noise magnitude")
    parser.add_argument("--uniform-low", type=float, default=-0.9, help="low bound for uniform noise")
    parser.add_argument("--uniform-high", type=float, default=0.9, help="high bound for uniform noise")
    parser.add_argument("--llm-disturb-interval", type=int, default=5, help="steps between disturbances")
    parser.add_argument("--num-test-episodes", type=int, default=800, help="number of testing episodes")

    # --- LLM-guided adversary ---
    parser.add_argument("--llm-guide", type=str, default="adversary", choices=["none", "adversary"],
                        help="enable LLM-guided perturbations")
    parser.add_argument("--llm-guide-type", type=str, default="stochastic",
                        choices=["stochastic", "uniform", "constraint"],
                        help="LLM adversarial perturbation type")
    
    # --- ERNIE regularization ---
    parser.add_argument("--use_ernie", action="store_true", default=False, help="If true, apply ERNIE regularization to policy updates")
    parser.add_argument("--lambda_ernie", type=float, default=0.01, help="Weight for the ERNIE adversarial regularization term in the policy loss.")
    parser.add_argument("--perturb_epsilon", type=float, default=0.001, help="Maximum magnitude of adversarial perturbation applied to observations.")
    parser.add_argument("--perturb_alpha", type=float, default=0.001, help="Step size (learning rate) for generating adversarial perturbations.")
    parser.add_argument("--perturb_num_steps", type=int, default=3, help="Number of gradient ascent steps used to generate adversarial perturbations.")

    # --- Diffusion settings ---
    parser.add_argument("--diffusion-horizon", type=int, default=25,
                        help="trajectory length H for diffusion model")
    parser.add_argument("--diffusion-steps", type=int, default=100,
                        help="number of diffusion steps T")
    parser.add_argument("--diffusion-batch-size", type=int, default=64)
    parser.add_argument("--diffusion-epochs", type=int, default=50)
    parser.add_argument("--diffusion-lr", type=float, default=1e-4)
    parser.add_argument("--diffusion-data-path", type=str, default="./diffusion_data.npz",
                        help="where to save/load (states,actions) trajectories")
    parser.add_argument("--diffusion-model-path", type=str, default="./diffusion_model.pt",
                        help="where to save the trained diffusion model")


    return parser.parse_args()



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
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
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


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        # print("=====obs_shape_n====", obs_shape_n)
        num_adversaries = min(env.n, arglist.num_adversaries)
        # print("=====num_adversary===", num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir, exp_name=arglist.exp_name)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        pp = 10

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # while pp>0:
            #     pp=pp-1
            #     for agent1, obs1 in zip(trainers,obs_n):
            #         print("====name===", agent1.name)
            #         print("==========action_n===========", agent1.action(obs1))
            #     print("===============================================")
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver, exp_name=arglist.exp_name)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # # saves final episode reward for plotting training curve later
            # if len(episode_rewards) > arglist.num_episodes:
            #     rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
            #     with open(rew_file_name, 'wb') as fp:
            #         pickle.dump(final_ep_rewards, fp)
            #     agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
            #     with open(agrew_file_name, 'wb') as fp:
            #         pickle.dump(final_ep_ag_rewards, fp)
            #     print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            #     break

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                # ensure directory exists
                # os.makedirs(os.path.dirname(arglist.plots_dir), exist_ok=True)
                os.makedirs(arglist.plots_dir, exist_ok=True)


                # prepare file paths
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.csv'
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.csv'

                # save overall mean rewards
                with open(rew_file_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["episode", "mean_reward"])   # header
                    for i, r in enumerate(final_ep_rewards, start=1):
                        writer.writerow([i * arglist.save_rate, r])

                # save per-agent rewards
                with open(agrew_file_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ["episode"] + ["agent_{}".format(i) for i in range(len(agent_rewards))]

                    writer.writerow(header)
                    for ep in range(len(final_ep_ag_rewards)//len(agent_rewards)):
                        row = [ (ep+1) * arglist.save_rate ]
                        for ag in range(len(agent_rewards)):
                            row.append(final_ep_ag_rewards[ep*len(agent_rewards)+ag])
                        writer.writerow(row)

                print("...Finished total of {} episodes. Saved CSV to {}".format(len(episode_rewards), rew_file_name))

                break


def train_multiple_runs(arglist, seed_list):
    """
    Run MADDPG multiple times with different seeds and save concatenated rewards
    in a single CSV. Includes per-agent rewards.
    """
    all_rewards = {}       # key: run_id, value: list of mean episode rewards
    all_agent_rewards = {} # key: run_id, value: list of lists (per agent)

    for run_id, seed in enumerate(seed_list):
        print("\n=== Starting run {} with seed {} ===".format(run_id, seed))

        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        tf.set_random_seed(seed)

        arglist.run_id = run_id
        arglist.seed = seed

        tf.reset_default_graph()   # reset TF graph
        max_mean_ep_reward = None

        with U.single_threaded_session():
            # Create environment
            env = make_env(arglist.scenario, arglist, arglist.benchmark)
            obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
            # print("=====obs_shape_n====", obs_shape_n)
            num_adversaries = min(env.n, arglist.num_adversaries)
            # print("=====num_adversary===", num_adversaries)
            trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
            print('Using good policy {} and adv policy {}'.format(arglist.good_policy, getattr(arglist, 'adv_policy', 'maddpg')))

            # Initialize variables
            U.initialize()

            # Load previous state if needed
            if arglist.load_dir == "":
                arglist.load_dir = arglist.save_dir
            if arglist.display or arglist.restore or arglist.benchmark:
                print('Loading previous state...')
                U.load_state(arglist.load_dir, exp_name=arglist.exp_name)

            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # per-agent rewards
            final_ep_rewards = []  # mean episode rewards for this run
            final_ep_ag_rewards = []  # per-agent rewards for this run
            agent_info = [[[]]]  # placeholder for benchmarking
            saver = tf.train.Saver()
            obs_n = env.reset()
            episode_step = 0
            train_step = 0
            t_start = time.time()

            # training loop
            while len(episode_rewards) <= arglist.num_episodes:
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)

                # store experience
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                if done or terminal:
                    obs_n = env.reset()
                    episode_step = 0
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])

                train_step += 1

                # update all trainers
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    agent.update(trainers, train_step)

                # save and log rewards
                if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                    U.save_state(arglist.save_dir, saver=saver, exp_name=arglist.exp_name)
                    mean_episode_reward = np.mean(episode_rewards[-arglist.save_rate:])
                    if max_mean_ep_reward is None or max_mean_ep_reward < mean_episode_reward:
                        max_mean_ep_reward = mean_episode_reward
                        U.save_state(arglist.save_dir, saver=saver, exp_name=arglist.exp_name+"best")
                    final_ep_rewards.append(mean_episode_reward)
                    per_agent_means = [np.mean(a[-arglist.save_rate:]) for a in agent_rewards]
                    final_ep_ag_rewards.append(per_agent_means)

                    print("Run {} | steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        run_id, train_step, len(episode_rewards), mean_episode_reward, per_agent_means, round(time.time()-t_start, 3)))
                    t_start = time.time()

        # store rewards for this run
        all_rewards[run_id] = final_ep_rewards
        all_agent_rewards[run_id] = final_ep_ag_rewards
        print("=== Finished run {} ===".format(run_id))

    # --- Save all runs to single CSV (mean rewards only) ---
    os.makedirs(arglist.plots_dir, exist_ok=True)
    exp_name = arglist.exp_name if arglist.exp_name is not None else "default_exp"
    csv_file = os.path.join(arglist.plots_dir, exp_name + "_all_runs_mean.csv")

    max_len = max(len(r) for r in all_rewards.values())

    # pad shorter runs
    for rid in all_rewards:
        if len(all_rewards[rid]) < max_len:
            all_rewards[rid] += [all_rewards[rid][-1]] * (max_len - len(all_rewards[rid]))

    # write CSV
    header = ["episode"] + ["run_{}".format(rid) for rid in sorted(all_rewards.keys())]

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(max_len):
            row = [(i+1) * arglist.save_rate]  # episode number
            for rid in sorted(all_rewards.keys()):
                row.append(all_rewards[rid][i])
            writer.writerow(row)

    print("Saved concatenated mean episode rewards for all runs to {}".format(csv_file))


def testWithoutP(arglist):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        #if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
        print('Loading trained model from {}'.format(arglist.load_dir))
        U.load_state(arglist.load_dir, exp_name=arglist.exp_name)

        # Parameters for testing
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len

        all_rewards = []
        print('Starting testing...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)
            for step in range(max_episode_len):
                # get actions from trained policies
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                new_obs_n, rew_n, done_n, _ = env.step(action_n)

                episode_reward += rew_n
                obs_n = new_obs_n

                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))


def testRobustnessOP(arglist):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        #if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
        print('Loading trained model from {}'.format(arglist.load_dir))
        U.load_state(arglist.load_dir, exp_name=arglist.exp_name)

        # Testing params
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()

            disrupted_obs_n = []
            for i, obs in enumerate(obs_n):
                disrupted_obs_n.append(apply_observation_disruption(
                    obs, 0, env, arglist
                ))

            obs_n = disrupted_obs_n

            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):
                # get actions
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)

                # === Apply your disruption here ===
                disrupted_obs_n = []
                # print("=================== before perturbation ===========")
                # print(new_obs_n)
                for i, obs in enumerate(new_obs_n):
                    disrupted_obs_n.append(apply_observation_disruption(
                        obs, rew_n[i], env, arglist
                    ))

                # print("=================== after perturbation ===========")
                # print(np.array(disrupted_obs_n) - np.array(new_obs_n))

                # track reward
                episode_reward += rew_n

                obs_n = disrupted_obs_n

                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))


def testRobustnessOA(arglist):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        arglist.load_dir = arglist.save_dir
        print('Loading trained model from {}'.format(arglist.load_dir))
        U.load_state(arglist.load_dir, exp_name=arglist.exp_name)

        # Testing params
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            disrupted_obs_n = []
            for i, obs in enumerate(obs_n):
                disrupted_obs_n.append(apply_observation_disruption(
                    obs, 0, env, arglist
                ))

            obs_n = disrupted_obs_n
            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):
                # --- Apply observation disruption before action selection ---
                obs_n_disrupted = [
                    apply_observation_disruption(obs, 0, env, arglist)
                    for obs in obs_n
                ]

                # --- Get actions from agents ---
                action_n = [
                    agent.action(obs_dis)
                    for agent, obs_dis in zip(trainers, obs_n_disrupted)
                ]

                # --- Apply action disruption ---
                action_n_disrupted = [
                    apply_action_disruption(action, 0, env, arglist)
                    for action in action_n
                ]

                # --- Environment step ---
                new_obs_n, rew_n, done_n, info_n = env.step(action_n_disrupted)

                # --- Track reward ---
                episode_reward += rew_n
                obs_n = new_obs_n

                # --- Render if needed ---
                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))
    
def testRobustnessAP(arglist, deffusion=True):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)

        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print('Testing using good policy {} and adv policy {}'.format(
            arglist.good_policy, arglist.adv_policy))

        # Initialize TF graph
        U.initialize()

        # Load trained model
        arglist.load_dir = arglist.save_dir
        print('Loading trained model from {}'.format(arglist.load_dir))
        U.load_state(arglist.load_dir, exp_name=arglist.exp_name)
        load_diffusion_model(arglist)

        # Testing params
        n_episodes = arglist.num_test_episodes
        max_episode_len = arglist.max_episode_len
        all_rewards = []

        # --- Extra for disruption ---
        env.llm_disturb_iteration = 0
        env.previous_reward = 0

        print('Starting testing with robustness perturbations...')

        for ep in range(n_episodes):
            obs_n = env.reset()
            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):

                # --- Get actions from agents ---
                # action_n = [
                #     agent.action(obs_dis)
                #     for agent, obs_dis in zip(trainers, obs_n)
                # ]

                # # --- Apply action disruption ---
                # action_n_disrupted = [
                #     apply_action_disruption(action, 0, env, arglist)
                #     for action in action_n
                # ]

                # --- clean MADDPG actions ---
                action_n = [
                    agent.action(obs_dis)
                    for agent, obs_dis in zip(trainers, obs_n)
                ]
                # print("=======MADDPG actions:=========")
                # print(action_n)

                # Number of agents
                n_agents = len(action_n)

                # Action dimension per agent (assume all agents have same action dim)
                action_dim_per_agent = action_n[0].shape[0]

                # print("Number of agents:", n_agents)
                # print("Action dimension per agent:", action_dim_per_agent)

                # --- adversarial noise ---
                action_n_noisy = [
                    apply_action_disruption(action, 0, env, arglist)
                    for action in action_n
                ]
                # print("=======Noisy actions:=========")
                # print(action_n_noisy)
                action_n_clean = action_n_noisy
                if deffusion:
                    # --- diffusion denoising ---
                    action_vec_noisy = concat_actions(action_n_noisy)
                    # print("=======Noisy action vec:=========")
                    # print(action_vec_noisy)
                    state_vec = np.concatenate(obs_n, axis=0)

                    action_vec_clean = diffusion_denoise_action(
                        action_vec_noisy,
                        state_vec,
                        t_start=40
                    )
                    # print("=======Clean action vec:=========")
                    # print(action_vec_clean)

                    action_n_clean = split_actions(action_vec_clean, n_agents, action_dim_per_agent)
                # print("===========Clean actions:=============")
                # print(action_n_clean)

                # for i, a in enumerate(action_n_clean):
                #     print("Agent {}: type={}, len={}, inner shape={}".format(
                #         i, type(a), len(a), a[0].shape
                #     ))


                # --- env step ---
                new_obs_n, rew_n, done_n, info_n = env.step(action_n_clean)


                # --- Environment step ---
                # new_obs_n, rew_n, done_n, info_n = env.step(action_n_disrupted)

                # --- Track reward ---
                episode_reward += rew_n
                obs_n = new_obs_n

                # --- Render if needed ---
                if arglist.display:
                    env.render()
                    time.sleep(0.05)

                if all(done_n):
                    break

            all_rewards.append(episode_reward)
            # print("Episode {} reward (per agent): {}".format(ep + 1, episode_reward))

        mean_rewards = np.mean(all_rewards, axis=0)
        print("Average reward per agent over {} episodes: {}".format(n_episodes, mean_rewards))
        print("Average total reward: {}".format(np.mean(np.sum(all_rewards, axis=1))))
        return np.mean(np.sum(all_rewards, axis=1))

def collect_diffusion_data(arglist):
    """
    Roll out the trained MADDPG policy and collect (state, action) trajectories
    for diffusion training. Uses the *clean* environment (no adversarial noise).

    Saves a .npz file with:
        states:  [N, H, Ds]   (global state = concat of obs_n)
        actions: [N, H, Da]   (global action = concat of action_n)
    """
    tf.reset_default_graph()
    H = arglist.diffusion_horizon

    with U.single_threaded_session():
        # 1) Build env & trainers
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

        print("[Diffusion] Using trained MADDPG from {}".format(arglist.save_dir))
        U.initialize()
        # load your best or final MADDPG policy
        U.load_state(arglist.save_dir, exp_name=arglist.exp_name)

        # 2) Figure out global dims
        # state_dim = sum of per-agent obs dims
        state_dim = sum(int(np.prod(s)) for s in obs_shape_n)

        # action_dim = sum of per-agent action dims (handle Box/Discrete)
        action_dim = get_total_action_dim(env)

        print("[Diffusion] state_dim={}, action_dim={}, horizon={}".format(
            state_dim, action_dim, H))

        state_trajs = []
        action_trajs = []

        num_episodes = arglist.num_episodes  # how many episodes to use for data
        max_episode_len = arglist.max_episode_len

        print("[Diffusion] Collecting trajectories from MADDPG expert...")
        for ep in range(num_episodes):
            obs_n = env.reset()
            ep_states = []
            ep_actions = []

            for t in range(max_episode_len):
                # build global state
                state_vec = np.concatenate(obs_n, axis=0)  # [Ds]

                # get joint action from MADDPG
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                action_vec = np.concatenate(action_n, axis=0)  # [Da]

                ep_states.append(state_vec)
                ep_actions.append(action_vec)

                obs_n, rew_n, done_n, info_n = env.step(action_n)

                if all(done_n):
                    break

            ep_states = np.asarray(ep_states, dtype=np.float32)
            ep_actions = np.asarray(ep_actions, dtype=np.float32)

            # keep only episodes that are at least H long
            if ep_states.shape[0] < H:
                continue

            ep_states = ep_states[:H]
            ep_actions = ep_actions[:H]

            state_trajs.append(ep_states)
            action_trajs.append(ep_actions)

            if (ep + 1) % 50 == 0:
                print("[Diffusion] Collected {} episodes so far".format(ep + 1))

        states = np.stack(state_trajs, axis=0)   # [N,H,Ds]
        actions = np.stack(action_trajs, axis=0)  # [N,H,Da]
        print("[Diffusion] Final dataset: states {}, actions {}".format(
            states.shape, actions.shape))

        np.savez(arglist.diffusion_data_path, states=states, actions=actions)
        print("[Diffusion] Saved dataset to {}".format(arglist.diffusion_data_path))


class TrajectoryDiffusion(nn.Module):
    """
    Simple DDPM-style diffusion model for joint action trajectories.
    x: [B, H, Da]; cond: [B, Ds] (global state, here we use s_0)
    """
    def __init__(self, horizon, action_dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(horizon * action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, x_noisy, t, cond):
        """
        x_noisy: [B,H,Da]
        t      : [B] (0..T-1)
        cond   : [B,Ds]
        """
        B = x_noisy.shape[0]
        x_flat = x_noisy.reshape(B, -1)

        t_norm = t.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_mlp(t_norm)
        c_emb = self.cond_mlp(cond)
        h = t_emb + c_emb

        h_cat = torch.cat([x_flat, h], dim=-1)
        eps_pred = self.net(h_cat)
        eps_pred = eps_pred.view(B, self.horizon, self.action_dim)
        return eps_pred


def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def q_sample(x0, t, eps, alphas_bar):
    """
    Forward diffusion q(x_t | x_0)
    x0 : [B,H,Da]
    t  : [B]
    eps: [B,H,Da]
    """
    a_bar = alphas_bar[t].view(-1, 1, 1).to(x0.device)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps


def train_diffusion(arglist):
    """
    Train a diffusion model from the dataset created by collect_diffusion_data().
    """
    data = np.load(arglist.diffusion_data_path)
    states = data["states"]   # [N,H,Ds]
    actions = data["actions"] # [N,H,Da]

    N, H, Ds = states.shape
    _, H2, Da = actions.shape
    assert H == H2 == arglist.diffusion_horizon

    print("[Diffusion] Loaded dataset:", states.shape, actions.shape)

    # Force CPU to avoid CUDA / CUBLAS issues
    device = torch.device("cpu")
    print("[Diffusion] Forcing device to CPU")

    # Convert to tensors
    states_t = torch.from_numpy(states).float()
    actions_t = torch.from_numpy(actions).float()

    # Optional: simple normalization (you can save mean/std if you like)
    act_mean = actions_t.mean(dim=(0, 1), keepdim=True)
    act_std = actions_t.std(dim=(0, 1), keepdim=True) + 1e-6
    actions_t = (actions_t - act_mean) / act_std

    model = TrajectoryDiffusion(
        horizon=H,
        action_dim=Da,
        cond_dim=Ds,
        hidden_dim=256
    ).to(device)

    betas, alphas, alphas_bar = make_beta_schedule(arglist.diffusion_steps)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=arglist.diffusion_lr)
    batch_size = arglist.diffusion_batch_size
    num_batches = max(1, N // batch_size)

    print("[Diffusion] Training on {} trajectories".format(N))
    for epoch in range(arglist.diffusion_epochs):
        perm = torch.randperm(N)
        states_t = states_t[perm]
        actions_t = actions_t[perm]

        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = min(N, (b + 1) * batch_size)

            x0 = actions_t[start:end].to(device)          # [B,H,Da]
            cond = states_t[start:end, 0, :].to(device)   # condition on s_0

            B = x0.shape[0]
            t = torch.randint(0, arglist.diffusion_steps, (B,), device=device)
            eps = torch.randn_like(x0)

            x_t = q_sample(x0, t, eps, alphas_bar)
            eps_pred = model(x_t, t, cond)

            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * B

        epoch_loss /= N
        print("[Diffusion] Epoch {}/{} - loss {:.6f}".format(
            epoch + 1, arglist.diffusion_epochs, epoch_loss))

    # save model + normalization info
    os.makedirs(os.path.dirname(arglist.diffusion_model_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "horizon": H,
            "action_dim": Da,
            "cond_dim": Ds,
            "diffusion_steps": arglist.diffusion_steps,
            "act_mean": act_mean,
            "act_std": act_std,
        },
        arglist.diffusion_model_path,
    )
    print("[Diffusion] Saved model to {}".format(arglist.diffusion_model_path))

def load_diffusion_model(arglist):
    global DIFFUSION_MODEL, DIFFUSION_CONSTS

    ckpt = torch.load(arglist.diffusion_model_path, map_location="cpu")

    model = TrajectoryDiffusion(
        horizon=ckpt["horizon"],
        action_dim=ckpt["action_dim"],
        cond_dim=ckpt["cond_dim"],
        hidden_dim=256,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    betas, alphas, alphas_bar = make_beta_schedule(ckpt["diffusion_steps"])

    DIFFUSION_MODEL = model
    DIFFUSION_CONSTS = {
        "betas": betas,
        "alphas": alphas,
        "alphas_bar": alphas_bar,
        "act_mean": ckpt["act_mean"],
        "act_std": ckpt["act_std"],
        "T": ckpt["diffusion_steps"],
        "H": ckpt["horizon"],
    }

    print("[Diffusion] Loaded trained diffusion model")

@torch.no_grad()
def diffusion_denoise_action(
    noisy_action_vec,
    state_vec,
    t_start=40
):
    """
    noisy_action_vec: [Da]
    state_vec       : [Ds]
    returns clean_action_vec [Da]
    """
    model = DIFFUSION_MODEL
    C = DIFFUSION_CONSTS

    H = C["H"]
    betas = C["betas"]
    alphas = C["alphas"]
    alphas_bar = C["alphas_bar"]

    # ---- normalize noisy action ----
    a = torch.from_numpy(noisy_action_vec).float()
    a = (a - C["act_mean"][0, 0]) / C["act_std"][0, 0]

    # ---- build x_t ----
    x = torch.zeros((1, H, a.shape[0]))
    x[0, 0] = a

    cond = torch.from_numpy(state_vec).float().unsqueeze(0)

    # ---- reverse diffusion ----
    for t in reversed(range(t_start + 1)):
        t_tensor = torch.tensor([t])

        eps_pred = model(x, t_tensor, cond)

        alpha = alphas[t]
        alpha_bar = alphas_bar[t]

        x0_hat = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

        if t > 0:
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha) * x0_hat + torch.sqrt(1 - alpha) * noise
        else:
            x = x0_hat

    # ---- unnormalize ----
    clean = x[0, 0] * C["act_std"][0, 0] + C["act_mean"][0, 0]
    return clean.numpy()

def concat_actions(action_n):
    return np.concatenate(action_n, axis=0)

def split_actions(action_vec, n_agents, action_dim_per_agent):
    """
    Split a flat action vector into a list of per-agent actions.
    
    Args:
        action_vec: flat np.array of shape (n_agents * action_dim_per_agent,)
        n_agents: int, number of agents
        action_dim_per_agent: int, dimension of action for each agent
        
    Returns:
        List of np.arrays of shape (action_dim_per_agent,) for each agent
    """
    # assert len(action_vec) == n_agents * action_dim_per_agent, \
    #     f"Length mismatch: {len(action_vec)} != {n_agents}*{action_dim_per_agent}"
    
    split = []
    for i in range(n_agents):
        start = i * action_dim_per_agent
        end = start + action_dim_per_agent
        split.append(action_vec[start:end])
    return split






def apply_observation_disruption(observation, reward, env, args):
    obs_orig = np.array(observation, dtype=np.float32)

    # === Apply noise ===
    
    if args.noise_type == "gauss":
        noise = np.random.normal(0, 0.5, size=obs_orig.shape)
        # print(noise)
        obs_orig = obs_orig + noise
    elif args.noise_type == "shift":
        obs_orig = obs_orig + args.noise_shift
    elif args.noise_type == "uniform":
        noise = np.random.uniform(0, 0.2, size=obs_orig.shape)
        obs_orig = obs_orig + noise

        
    return obs_orig


def apply_action_disruption(action, reward, env, args):
    action_orig = np.array(action, dtype=np.float32)

    if args.noise_type == "gauss":
        action_orig = action_orig + np.random.normal(0, arglist.act_noise, size=action_orig.shape)
    elif args.noise_type == "shift":
        action_orig = action_orig + args.noise_shift
    elif args.noise_type == "uniform":
        action_orig = action_orig + np.random.uniform(1, 4, size=action_orig.shape)

    return action_orig

if __name__ == '__main__':
    arglist = parse_args()
    print(arglist.act_noise)

    if arglist.mode == "train":
        seed_list = [1, 2, 3, 4, 5]
        train_multiple_runs(arglist, seed_list)

    elif arglist.mode == "test":
        # rew = testWithoutP(arglist)
        # act_noise_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # for noise in act_noise_list:
        #     arglist.act_noise = noise
        #     print("Reward without any noise {}".format(rew))
        #     rew = testRobustnessAP(arglist, deffusion=False)
        #     print("Reward with noise shift {}: {}".format(noise, rew))
        #     rew = testRobustnessAP(arglist)
        #     print("Reward with noise shift with deffusion {}: {}".format(noise, rew))

        csv_filename = "{}_robustness_results.csv".format(arglist.exp_name)

        # Prepare results list
        results = []

        # Baseline (no noise, no diffusion)
        rew_no_noise = testWithoutP(arglist)
        print("Reward without any noise {}".format(rew_no_noise))

        # Noise levels
        act_noise_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]

        for noise in act_noise_list:
            arglist.act_noise = noise

            # Without diffusion
            rew_no_diff = testRobustnessAP(arglist, deffusion=False)
            print("Reward with noise shift {}: {}".format(noise, rew_no_diff))

            # With diffusion
            rew_with_diff = testRobustnessAP(arglist)
            print("Reward with noise shift with diffusion {}: {}".format(noise, rew_with_diff))

            # Store row
            results.append([
                noise,
                rew_no_noise,
                rew_no_diff,
                rew_with_diff
            ])

        # Write CSV
        with open(csv_filename, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow([
                "action_noise_std",
                "reward_no_noise",
                "reward_noise_no_diffusion",
                "reward_noise_with_diffusion"
            ])
            writer.writerows(results)

        print("Saved robustness results to {}".format(csv_filename))

        # all_results = []

        # for run_id in range(arglist.num_test_runs):
        #     print("\n================ Run {}/{} ================".format(
        #         run_id + 1, arglist.num_test_runs))
        #     run_results = {"run": run_id + 1}

        #     # baseline (no noise)
        #     rew = testWithoutP(arglist)
        #     run_results["none"] = rew

        #     for noise in ["gauss"]:
        #         arglist.noise_type = noise

        #         rew = testRobustnessOP(arglist)
        #         run_results["{}_obs_only".format(noise)] = rew

        #         rew = testRobustnessAP(arglist)
        #         run_results["{}_act_only".format(noise)] = rew

        #         rew = testRobustnessOA(arglist)
        #         run_results["{}_obs+action".format(noise)] = rew

        #     all_results.append(run_results)

        # df = pd.DataFrame(all_results)
        # exp_name = arglist.exp_name if arglist.exp_name is not None else "default_exp"
        # df.to_csv(exp_name + "_test_rewards.csv", index=False)

    elif arglist.mode == "collect_diffusion":
        collect_diffusion_data(arglist)

    elif arglist.mode == "train_diffusion":
        train_diffusion(arglist)

    
