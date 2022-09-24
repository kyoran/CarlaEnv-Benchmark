# !/usr/bin/python3
# -*- coding: utf-8 -*-

import gym
import numpy as np
from collections import deque

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._dvs_frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        #         obs = self.env.reset()
        #         for _ in range(self._k):
        #             self._frames.append(obs)
        #         return self._get_obs()
        obs = self.env.reset()
        for _ in range(self._k):
            self._dvs_frames.append(obs["dvs_frame"])

        dvs_stack_frames = self._get_stack_events()
        obs.update({
            'dvs_stack_frames': dvs_stack_frames
        })
        return obs

    def step(self, action):
        #         obs, reward, done, info = self.env.step(action)
        #         self._frames.append(obs)
        #         return self._get_obs(), reward, done, info
        obs, reward, done, info = self.env.step(action)
        self._dvs_frames.append(obs["dvs_frame"])
        dvs_stack_frames = self._get_stack_events()
        obs.update({
            'dvs_stack_frames': dvs_stack_frames
        })
        return obs, reward, done, info

    def _get_stack_events(self):
        assert len(self._dvs_frames) == self._k
        return np.concatenate(list(self._dvs_frames), axis=0)