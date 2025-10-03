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
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
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

    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Run mode: 'train' to train agents, 'test' to run evaluation")
    parser.add_argument("--num_test_runs", type=int, default=5, help="Number of test runs to perform when in test mode")



    # --- Robustness settings ---
    parser.add_argument("--noise-factor", type=str, default="state", choices=["none", "state", "reward"],
                        help="where to apply noise (state/reward/none)")
    parser.add_argument("--noise-type", type=str, default="gauss", choices=["gauss", "shift", "uniform"],
                        help="type of noise distribution")
    parser.add_argument("--noise-mu", type=float, default=0.0, help="mean for Gaussian noise")
    parser.add_argument("--noise-sigma", type=float, default=1, help="std for Gaussian noise")
    parser.add_argument("--noise-shift", type=float, default=0.05, help="shift noise magnitude")
    parser.add_argument("--uniform-low", type=float, default=-0.1, help="low bound for uniform noise")
    parser.add_argument("--uniform-high", type=float, default=0.1, help="high bound for uniform noise")
    parser.add_argument("--llm-disturb-interval", type=int, default=5, help="steps between disturbances")
    parser.add_argument("--num-test-episodes", type=int, default=1000, help="number of testing episodes")

    # --- LLM-guided adversary ---
    parser.add_argument("--llm-guide", type=str, default="adversary", choices=["none", "adversary"],
                        help="enable LLM-guided perturbations")
    parser.add_argument("--llm-guide-type", type=str, default="stochastic",
                        choices=["stochastic", "uniform", "constraint"],
                        help="LLM adversarial perturbation type")


    return parser.parse_args()

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
        print("=====obs_shape_n====", obs_shape_n)
        num_adversaries = min(env.n, arglist.num_adversaries)
        print("=====num_adversary===", num_adversaries)
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
            print("=====obs_shape_n====", obs_shape_n)
            num_adversaries = min(env.n, arglist.num_adversaries)
            print("=====num_adversary===", num_adversaries)
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
    
def testRobustnessAP(arglist):
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
            episode_reward = np.zeros(env.n)

            for step in range(max_episode_len):

                # --- Get actions from agents ---
                action_n = [
                    agent.action(obs_dis)
                    for agent, obs_dis in zip(trainers, obs_n)
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


def apply_observation_disruption(observation, reward, env, args):
    """
    Apply noise or LLM-guided disruption to observation/reward.
    Ensures the output is always a valid NumPy array.
    """
    # Keep track of iterations
    env.llm_disturb_iteration += 1
    obs_orig = np.array(observation, dtype=np.float32)

    # === Apply noise ===
    if args.noise_factor == "state" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        if args.noise_type == "gauss":
            noise = np.random.normal(args.noise_mu, args.noise_sigma, size=obs_orig.shape)
            # print(noise)
            noise = 10*noise+5
            obs_orig = obs_orig + noise
        elif args.noise_type == "shift":
            obs_orig = obs_orig + args.noise_shift
        elif args.noise_type == "uniform":
            noise = np.random.uniform(args.uniform_low, args.uniform_high, size=obs_orig.shape)
            noise = 5*noise
            obs_orig = obs_orig + noise

    # === LLM-guided perturbation ===
    if args.llm_guide == "adversary" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        prompt = (
            "Robust RL adversary: Current reward = {}, previous reward = {}. "
            "Revise this state: {}. Output only the revised state as a Python list."
        ).format(reward, env.previous_reward, obs_orig.tolist())

        obs_from_gpt = gpt_call(prompt)
        if obs_from_gpt is not None:
            try:
                obs_from_gpt = np.array(eval(obs_from_gpt), dtype=np.float32)
                # reshape/pad/truncate to match original observation
                if obs_from_gpt.shape != obs_orig.shape:
                    flat = obs_from_gpt.flatten()
                    size_needed = np.prod(obs_orig.shape)
                    if flat.size < size_needed:
                        flat = np.pad(flat, (0, size_needed - flat.size), mode="constant")
                    else:
                        flat = flat[:size_needed]
                    obs_from_gpt = flat.reshape(obs_orig.shape)
                obs_orig = obs_from_gpt
            except:
                # fallback to original observation
                pass

    env.previous_reward = reward
    return obs_orig


def apply_action_disruption(action, reward, env, args):
    env.llm_disturb_iteration += 1
    action_orig = np.array(action, dtype=np.float32)

    if args.noise_factor == "action" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        if args.noise_type == "gauss":
            action_orig = action_orig + np.random.normal(args.noise_mu, args.noise_sigma, size=action_orig.shape)
        elif args.noise_type == "shift":
            action_orig = action_orig + args.noise_shift
        elif args.noise_type == "uniform":
            action_orig = action_orig + np.random.uniform(args.uniform_low, args.uniform_high, size=action_orig.shape)

    if args.llm_guide == "adversary" and env.llm_disturb_iteration % args.llm_disturb_interval == 0:
        prompt = (
            "Robust RL adversary: Current reward = {}, previous reward = {}. "
            "Revise this action: {}. Output only the revised action as a Python list."
        ).format(reward, env.previous_reward, action_orig.tolist())
        action_from_gpt = gpt_call(prompt)
        if action_from_gpt is not None:
            try:
                action_from_gpt = np.array(eval(action_from_gpt), dtype=np.float32)
                if action_from_gpt.shape != action_orig.shape:
                    flat = action_from_gpt.flatten()
                    size_needed = np.prod(action_orig.shape)
                    if flat.size < size_needed:
                        flat = np.pad(flat, (0, size_needed - flat.size), mode="constant")
                    else:
                        flat = flat[:size_needed]
                    action_from_gpt = flat.reshape(action_orig.shape)
                action_orig = action_from_gpt
            except:
                pass

    env.previous_reward = reward
    return action_orig

if __name__ == '__main__':
    arglist = parse_args()
    # train(arglist)
    if arglist.mode == "train":
        seed_list = [1]  # list of random seeds for multiple runs
        train_multiple_runs(arglist, seed_list)
    else:
        all_results = []

        for run_id in range(arglist.num_test_runs):
            print("\n================ Run {}/{} ================".format(run_id + 1, arglist.num_test_runs))
            run_results = {"run": run_id + 1}

            # baseline (no noise)
            rew = testWithoutP(arglist)
            run_results["none"] = rew

            for noise in ["gauss", "shift", "uniform"]:
                arglist.noise_type = noise

                rew = testRobustnessOP(arglist)
                run_results["{}_obs_only".format(noise)] = rew

                rew = testRobustnessAP(arglist)
                run_results["{}_act_only".format(noise)] = rew

                rew = testRobustnessOA(arglist)
                run_results["{}_obs+action".format(noise)] = rew

            all_results.append(run_results)

        # convert to dataframe
        df = pd.DataFrame(all_results)

        # save to CSV
        exp_name = arglist.exp_name if arglist.exp_name is not None else "default_exp"
        df.to_csv(exp_name +"_test_rewards.csv", index=False)

        # print("\n✅ Saved test results for", arglist.num_runs, "runs to test_rewards.csv")
    
