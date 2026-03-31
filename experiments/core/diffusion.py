"""
Diffusion model utilities for denoising actions.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= Diffusion globals =================
DIFFUSION_MODEL = None
DIFFUSION_CONSTS = {}
DIFFUSION_DEVICE = torch.device("cpu")


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
    """
    Create beta schedule for diffusion process.
    """
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


def collect_diffusion_data(arglist):
    """
    Roll out the trained MADDPG policy and collect (state, action) trajectories
    for diffusion training. Uses the *clean* environment (no adversarial noise).

    Saves a .npz file with:
        states:  [N, H, Ds]   (global state = concat of obs_n)
        actions: [N, H, Da]   (global action = concat of action_n)
    """
    from .environment import make_env, get_trainers, get_total_action_dim
    import maddpg.common.tf_util as U
    import tensorflow as tf

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
    import os
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
    """
    Load trained diffusion model and constants.
    """
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
    """
    Concatenate list of per-agent actions into single vector.
    """
    return np.concatenate(action_n, axis=0)


def split_actions(action_vec, n_agents, action_dim_per_agent):
    """
    Split a flat action vector into a list of per-agent actions.
    """
    split = []
    start = 0
    for dim in action_dim_per_agent:
        end = start + dim
        split.append(action_vec[start:end])
        start = end
    return split