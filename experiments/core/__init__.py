"""
Core modules for MADDPG experiments.
"""

from .config import parse_args
from .environment import make_env, get_trainers, get_total_action_dim, mlp_model
from .training import train, train_multiple_runs
from .testing import testWithoutP, testRobustnessOP, testRobustnessOA, testRobustnessAP
from .noise import apply_observation_disruption, apply_action_disruption
from .diffusion import (
    collect_diffusion_data, train_diffusion, load_diffusion_model,
    diffusion_denoise_action, concat_actions, split_actions
)

__all__ = [
    # Config
    'parse_args',
    # Environment
    'make_env', 'get_trainers', 'get_total_action_dim', 'mlp_model',
    # Training
    'train', 'train_multiple_runs',
    # Testing
    'testWithoutP', 'testRobustnessOP', 'testRobustnessOA', 'testRobustnessAP',
    # Noise
    'apply_observation_disruption', 'apply_action_disruption',
    # Diffusion
    'collect_diffusion_data', 'train_diffusion', 'load_diffusion_model',
    'diffusion_denoise_action', 'concat_actions', 'split_actions'
]