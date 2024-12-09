import torch
import numpy as np


def evaluate(
    envs,
    model_path: str,
    eval_episodes: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
):

    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    print("Starting evaluation")
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "episode" in infos.keys():
            episode_info = infos["episode"]
            ep_returns = episode_info["r"]
            ep_completions = episode_info["_r"]
            mean_returns = np.mean(ep_returns[ep_completions])
            print(
                f"eval_episode={len(episodic_returns)}, episodic_return={mean_returns}"
            )
            episodic_returns += [mean_returns]
        obs = next_obs

    return episodic_returns
