import gymnasium as gym

import numpy as np


class BoxTargetEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.max_episode_steps = 20

        self.done = False
        self.t = 0
        self.goal_position = np.array([0.0, 0.0, 0.0])
        self.agent_position = np.array([0.0, 0.0, 0.0])
        self.reset()

    def reset(self, seed=None, options=None):
        self.done = False
        self.t = 0
        self.goal_position = np.random.uniform(-1.0, 1.0, size=3)
        self.agent_position = np.random.uniform(-1.0, 1.0, size=3)

        return self.observation(), {}

    def move_agent(self, action):
        displacement = np.clip(action, -1, 1) * 2.0
        self.agent_position += displacement
        self.agent_position = np.clip(self.agent_position, -1.0, 1.0)

    def observation(self):
        return np.concatenate([self.agent_position, self.goal_position])

    def get_distance(self):
        return np.linalg.norm(self.agent_position - self.goal_position)

    def get_reward(self):
        return -self.get_distance()

    def step(self, action):
        self.t += 1
        self.move_agent(action)
        if self.t >= self.max_episode_steps:
            self.done = True

        info = {
            "distance": self.get_distance(),
        }

        return self.observation(), self.get_reward(), self.done, False, info
