import functools
from collections import deque

import gym
import jax
import numpy as np


class BasicWrapper(gym.Wrapper):

    def __init__(self, env):
        super(BasicWrapper, self).__init__(env)
        self.observation_shape = (120, 160, 2)
        self.step_count = 0
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)['_next_state']['STEREO_CAMERAS']
    def step(self, action: int):
        self.step_count += 1
        observation, reward, done, info = self.env.step(action)
        return observation['STEREO_CAMERAS'], reward, done, info

class BasicWrapperOneChannel(gym.Wrapper):

    def __init__(self, env):
        super(BasicWrapperOneChannel, self).__init__(env)
        self.observation_shape = (120, 160, 1)
        self.step_count = 0
    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)['_next_state']['STEREO_CAMERAS'][:, :, :1]
    def step(self, action: int):
        self.step_count += 1
        observation, reward, done, info = self.env.step(action)
        return observation['STEREO_CAMERAS'][:, :, :1], reward, done, info

class FrameStackB(gym.Wrapper):
    """
    Shape of obs: (120, 160, 2*num_stack)
    """
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super(FrameStackB, self).__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        w, h, c = self.observation_shape
        self.observation_shape = w, h, c*num_stack
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()
    def step(self, action: int):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info
    def observation(self):
        #return jnp.concatenate(self.frames, -1)
        return concat(self.frames)


def concat(frames):
    out = np.concatenate(frames, -1)
    return out


class FrameSkipMax(gym.Wrapper):
    """
    Shape of obs: (120, 160, 2*num_stack)
    """
    def __init__(self, env: gym.Env, num_skip: int = 2):
        super(FrameSkipMax, self).__init__(env)
        self.num_skip = num_skip
        self._obs_buffer = np.zeros((2,) + self.observation_shape)
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self.num_skip):
            obs, reward, done, info = self.env.step(action)
            if i == self.num_skip - 2:
                self._obs_buffer[0] = obs
            if i == self.num_skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

class FrameSkip(gym.Wrapper):
    """
    Shape of obs: (120, 160, 2*num_stack)
    """
    def __init__(self, env: gym.Env, num_skip: int = 2, max_buffer=False):
        super(FrameSkip, self).__init__(env)
        self.num_skip = num_skip
        self.max_buffer = max_buffer
        if max_buffer:
            self._obs_buffer = np.zeros((2,) + self.observation_shape)
        
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self.num_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class ReshapeWrapperJit(gym.Wrapper):
    """
    Only one dimension subsracted from obs!
    """
    def __init__(self, env, shape=(64, 64, 1)):
        super(ReshapeWrapperJit, self).__init__(env)
        self.observation_shape = shape
        self.reshape_function = jax.jit(functools.partial(reshape, shape=shape))
    def reset(self, **kwargs):
        return self.reshape_function(self.env.reset(**kwargs))
    def step(self, action:int):
        observation, reward, done, info = self.env.step(action)
        return self.reshape_function(observation), reward, done, info

def reshape(observation, shape):
    out = jax.image.resize(observation, shape=shape, method='bilinear')
    return out
