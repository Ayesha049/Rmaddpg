"""
Training functions for MADDPG experiments.
"""
import numpy as np
import tensorflow as tf
import time
import csv
import os

import maddpg.common.tf_util as U
from .environment import make_env, get_trainers


def train(arglist):
    """
    Single-run MADDPG training function.
    """
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
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

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
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
                        import pickle
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

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                # ensure directory exists
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
        import random
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
            num_adversaries = min(env.n, arglist.num_adversaries)
            trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)

            print('Using good policy {} and adv policy {}'.format(
                arglist.good_policy, getattr(arglist, 'adv_policy', 'maddpg')))

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