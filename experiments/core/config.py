"""
Configuration and argument parsing for MADDPG experiments.
"""
import argparse


def parse_args():
    """
    Parse command line arguments for MADDPG experiments.
    """
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
    parser.add_argument("--save-dir", type=str, default="../../models", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./model", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="../../results/", help="directory where plot data is saved")

    parser.add_argument("--run-id", type=int, default=0, help="ID of the run for multiple seeds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Run mode
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
    parser.add_argument("--num-test-episodes", type=int, default=80, help="number of testing episodes")

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
    parser.add_argument("--diffusion-steps", type=int, default=100, help="number of diffusion steps T")
    parser.add_argument("--diffusion-batch-size", type=int, default=64)
    parser.add_argument("--diffusion-epochs", type=int, default=50)
    parser.add_argument("--diffusion-lr", type=float, default=1e-4)
    parser.add_argument("--diffusion-data-path", type=str, default="../../diffusion_data.npz",
                        help="where to save/load (states,actions) trajectories")
    parser.add_argument("--diffusion-model-path", type=str, default="../../diffusion_model.pt",
                        help="where to save the trained diffusion model")

    return parser.parse_args()