"""
Noise and perturbation utilities for robustness testing.
"""
import numpy as np


def apply_observation_disruption(observation, reward, env, args):
    """
    Apply noise to observations for robustness testing.
    """
    obs_orig = np.array(observation, dtype=np.float32)

    # === Apply noise ===
    if args.noise_type == "gauss":
        noise = np.random.normal(0, 0.5, size=obs_orig.shape)
        obs_orig = obs_orig + noise
    elif args.noise_type == "shift":
        obs_orig = obs_orig + args.noise_shift
    elif args.noise_type == "uniform":
        noise = np.random.uniform(0, 0.2, size=obs_orig.shape)
        obs_orig = obs_orig + noise

    return obs_orig


def apply_action_disruption(action, reward, env, args):
    """
    Apply noise to actions for robustness testing.
    """
    action_orig = np.array(action, dtype=np.float32)

    if args.noise_type == "gauss":
        action_orig = action_orig + np.random.normal(args.noise_mu, args.act_noise, size=action_orig.shape)
    elif args.noise_type == "shift":
        action_orig = action_orig + args.noise_shift
    elif args.noise_type == "uniform":
        action_orig = action_orig + np.random.uniform(args.uniform_low, args.uniform_high, size=action_orig.shape)

    return action_orig