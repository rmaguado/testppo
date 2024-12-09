import os
import random
import time
import omegaconf
import argparse

from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from testenv import BoxTargetEnvironment
from ppo_eval import evaluate


def get_argparser():
    parser = argparse.ArgumentParser(description="Test PPO", add_help=True)
    parser.add_argument(
        "--config-file",
        type=str,
        default="config.yaml",
        help="path to the config file",
    )
    parser.add_argument("--output-dir", type=str, help="path to the output directory")
    return parser


def make_env(gamma):
    def thunk():
        # todo: test removing wrappers
        env = BoxTargetEnvironment()
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, feature_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), feature_dim
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(), feature_dim
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, feature_dim)),
            nn.Tanh(),
            layer_init(
                nn.Linear(feature_dim, np.prod(envs.single_action_space.shape)),
                std=0.01,
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def log_episode_statistics(writer, global_step, infos):
    episode_info = infos["episode"]
    ep_distance = infos["distance"]
    ep_returns = episode_info["r"]
    ep_lengths = episode_info["l"]
    ep_completions = episode_info["_r"]
    mean_returns = np.mean(ep_returns[ep_completions])
    mean_lengths = np.mean(ep_lengths[ep_completions])
    mean_distance = np.mean(ep_distance[ep_completions])
    writer.add_scalar("charts/episodic_return", mean_returns, global_step)
    writer.add_scalar("charts/episodic_length", mean_lengths, global_step)
    writer.add_scalar("charts/episodic_distance", mean_distance, global_step)


def train(cfg, writer, device):
    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(cfg.gamma) for i in range(cfg.num_envs)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs, cfg.feature_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    iterator = range(1, cfg.num_iterations + 1)
    if cfg.use_tqdm:
        iterator = tqdm(iterator)

    for iteration in iterator:
        # Annealing the rate if instructed to do so.
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            lrnow = frac * cfg.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "episode" in infos.keys():
                log_episode_statistics(writer, global_step, infos)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        lr = optimizer.param_groups[0]["lr"]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", lr, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    envs.close()
    return agent


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    config_path = args.config_file
    output_dir = args.output_dir

    cfg = omegaconf.OmegaConf.load(config_path)
    cfg.batch_size = int(cfg.num_envs * cfg.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.num_minibatches)
    cfg.num_iterations = cfg.total_timesteps // cfg.batch_size
    run_name = f"{cfg.exp_name}_{int(time.time())}"

    os.makedirs(f"{output_dir}/{run_name}", exist_ok=True)
    omegaconf.OmegaConf.save(cfg, f"{output_dir}/{run_name}/config.yaml")

    writer = SummaryWriter(f"{output_dir}/tensorboard/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    if cfg.cuda:
        assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda" if cfg.cuda else "cpu")
    print("Using", device)

    agent = train(cfg, writer, device)

    if cfg.save_model:
        os.makedirs(f"{output_dir}/models/{run_name}", exist_ok=True)
        model_path = f"{output_dir}/models/{run_name}/{cfg.exp_name}.model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            gym.vector.SyncVectorEnv([make_env(cfg.gamma)]),
            model_path=model_path,
            eval_episodes=10,
            Model=Agent,
            device=device,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    writer.close()
