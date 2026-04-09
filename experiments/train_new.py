"""
Main entry point for MADDPG experiments.
Organized into modular components in the core/ directory.
"""
import os
import csv
import pandas as pd
import numpy as np
import random

from core import (
    parse_args, train_multiple_runs, testWithoutP, testRobustnessAP,
    collect_diffusion_data, train_diffusion
)


def r2(x):
    """Format number to 2 decimal places."""
    return "{:.2f}".format(float(x))


if __name__ == '__main__':
    arglist = parse_args()
    # Set plots directory based on scenario - make it absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from experiments/ to Rmaddpg/
    arglist.plots_dir = os.path.join(project_root, "results", arglist.scenario) + "/"
    print("DEBUG: __file__ = {}".format(__file__))
    print("DEBUG: script_dir = {}".format(script_dir))
    print("DEBUG: project_root = {}".format(project_root))
    print("DEBUG: final plots_dir = {}".format(arglist.plots_dir))
    # Create the plots directory for all modes
    os.makedirs(arglist.plots_dir, exist_ok=True)
    # Ensure diffusion data directory exists
    os.makedirs(os.path.dirname(arglist.diffusion_data_path), exist_ok=True)
    print(arglist.act_noise)

    if arglist.mode == "train":
        seed_list = [1]
        train_multiple_runs(arglist, seed_list)

    elif arglist.mode == "test":
        # Comprehensive robustness testing with noise sweeps
        arglist.noise_type = "gauss"
        # Sweep settings
        noise_mu_list = [-0.5, 0.5]
        # act_std_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
        act_std_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        t_start_list = [20, 40]

        # Baseline (no noise, no diffusion)
        rew_no_noise = testWithoutP(arglist)
        print("Baseline (no noise): {:.3f}".format(rew_no_noise))

        # Create separate CSV files for each noise_mu value
        for noise_mu in noise_mu_list:
            arglist.noise_mu = noise_mu
            print("\n=== Noise mean = {} ===".format(noise_mu))

            csv_filename = "{}_gauss_mu_{}_actstd_tstart_sweep.csv".format(arglist.exp_name, r2(noise_mu).replace('.', 'p').replace('-', 'n'))
            print("DEBUG: plots_dir = {}".format(arglist.plots_dir))
            print("DEBUG: csv_filename = {}".format(csv_filename))
            print("DEBUG: full path = {}".format(os.path.join(arglist.plots_dir, csv_filename)))
            results = []

            for act_std in act_std_list:
                arglist.act_noise = act_std
                print("\n  === Action noise std = {} ===".format(act_std))

                # Noise, no diffusion
                rew_no_diff = testRobustnessAP(
                    arglist,
                    deffusion=False
                )

                print("    No diffusion reward: {:.3f}".format(rew_no_diff))

                # Store diffusion rewards per t_start
                diff_rewards = {}

                for t_start in t_start_list:
                    print("    -> t_start = {}".format(t_start))

                    rew_with_diff = testRobustnessAP(
                        arglist,
                        deffusion=True,
                        t_start=t_start
                    )

                    diff_rewards[t_start] = rew_with_diff

                    print(
                        "       with diffusion (t_start={}): {:.3f}".format(
                            t_start, rew_with_diff
                        )
                    )

                # Derived metrics
                best_diff_reward = max(diff_rewards.values())

                pct_inc_vs_no_diff = (
                    (best_diff_reward - rew_no_diff) / abs(rew_no_diff)
                ) * 100.0

                pct_inc_vs_no_noise = (
                    (best_diff_reward - rew_no_noise) / abs(rew_no_noise)
                ) * 100.0

                # Assemble row (exclude noise_mu since it's in filename)
                row = [
                    r2(act_std),
                    r2(rew_no_noise),
                    r2(rew_no_diff)
                ]

                for t_start in t_start_list:
                    row.append(r2(diff_rewards[t_start]))

                row.extend([
                    r2(best_diff_reward),
                    r2(pct_inc_vs_no_diff),
                    r2(pct_inc_vs_no_noise)
                ])

                results.append(row)

            # Dynamic CSV header (exclude noise_mu since it's in filename)
            header = [
                "action_noise_std",
                "reward_no_noise",
                "reward_noise_no_diffusion"
            ]

            for t_start in t_start_list:
                header.append("reward_with_diff_t{}".format(t_start))

            header.extend([
                "best_reward_with_diffusion",
                "pct_inc_vs_no_diffusion",
                "pct_inc_vs_no_noise_worst"
            ])

            with open(os.path.join(arglist.plots_dir, csv_filename), mode="w", newline="") as f:
                print("Saving CSV to: {}".format(os.path.abspath(os.path.join(arglist.plots_dir, csv_filename))))
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(results)

            print("Saved robustness results to {}".format(csv_filename))

    elif arglist.mode == "collect_diffusion":
        collect_diffusion_data(arglist)

    elif arglist.mode == "train_diffusion":
        train_diffusion(arglist)