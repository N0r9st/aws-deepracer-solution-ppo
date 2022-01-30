import functools
import pickle
import sys
from collections import deque
from time import time
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np

from env_deepracer import DeepracerGymEnv
from run_ppo_train import (ACLight, create_train_state, get_initial_params,
                           load_state)

Array = Any

def create_state(config):

    key = jax.random.PRNGKey(0)

    #model = ActorCritic(num_outputs=5)
    model = ACLight()
    params = get_initial_params(key, model, obs_shape=config.obs_shape)

    iterations_per_step = config.actor_steps// config.batch_size
    loop_steps = config.total_frames // config.actor_steps

    state = create_train_state(params=params, model=model, learning_rate=config.learning_rate, 
                        decaying_lr=config.decaying_lr_and_clip_param, train_steps=loop_steps * config.num_epochs * iterations_per_step)

    state = load_state(config.path, state, lin_scheduler=config.decaying_lr_and_clip_param)
    return state



class DeepracerAgent():
    def __init__(self):
        self.agent_type = None

    def register_reset(self, observations):
        raise NotImplementedError

    def compute_action(self, observations, info):
        raise NotImplementedError


class MyAgent(DeepracerAgent):
    def __init__(self, config):
        self.num_stack = 4
        self.num_skip = 1


        self.frames = deque(maxlen=self.num_stack)
        self.step_count = 0
        self.last_action = 2
        self.state = create_state(config)
        self.apply_network = jax.jit(functools.partial(get_action, apply_fn=self.state.apply_fn, params=self.state.params))
        self.image_shape = config.image_shape

    def append_frames(self, observations):
        observations = self.process_observations(observations,)
        self.frames.append(observations)

    def process_observations(self, observations):
        observations = observations['STEREO_CAMERAS'][:, :, :1]
        observations = jax.image.resize(observations, shape=self.image_shape + (1,), method='bilinear')
        return observations

    def register_reset(self, observations):
        if '_next_state' in observations:
            processed = self.process_observations(observations['_next_state'])
            [self.frames.append(processed) for _ in range(self.num_stack)]
        else:
            processed = self.process_observations(observations)
            [self.frames.append(processed) for _ in range(self.num_stack)]
        self.step_count = 1
        in_data = jnp.concatenate(self.frames, -1)
        action = self.apply_network(in_data).item()
        self.last_action = action
        return action

    def compute_action(self, observations, info):
        if self.step_count==0:
             [self.frames.append(self.process_observations(observations)) for _ in range(self.num_stack)]

        if self.step_count % self.num_skip==0:
            self.append_frames(observations)
            in_data = jnp.concatenate(self.frames, -1)
            action = self.apply_network(in_data).item()
            self.last_action = action
        else:
            action = self.last_action

        self.step_count += 1
        return action

def get_action(obs: Array, apply_fn: Callable, params: flax.core.FrozenDict):
    log_prob, _ = apply_fn({'params': params}, obs[None])
    action = np.argmax(log_prob)
    return action

def test_class(agent):
    print('Connecting to env...')
    env = DeepracerGymEnv(port=8887)
    _ = env.reset()
    print('Env connected!')
    run_count = 1
    total_reward = 0
    reward_list = []
    decision_times = []
    observations_list = []
    for _ in range(run_count):
        done = False
        observations = env.reset()

        sta = time()
        action = agent.register_reset(observations)
        decision_times.append((time()-sta))
        
        observations, reward, done, info = env.step(action)
        
        reward_list.append(reward)
        
        while not done:
            sta = time()
            action = agent.compute_action(observations, info)
            decision_times.append((time()-sta))
            observations, reward, done, info = env.step(action)
            
            observations_list.append(observations['STEREO_CAMERAS'][...])
            total_reward += reward
            reward_list.append(reward)
    print(f'MEAN REWARD OF {run_count} RUNS: {total_reward/run_count}')
    mean_ = np.mean(decision_times)
    median_ = np.median(decision_times)
    std_ = np.std(decision_times)
    min_ = min(decision_times)
    max_ = max(decision_times)
    argmax_ = np.argmax(decision_times)
    print(f'INFERENCE TIMES: mean={mean_}, median={median_}, min={min_}, max_={max_}, std={std_}, argmax={argmax_}')
    print('TOTAL STEPS', len(reward_list))
    print('MEAN REWARD', np.mean(reward_list))

    from PIL import Image
    img, *imgs = [Image.fromarray(observation) for observation in observations_list]
    img.save(fp='gif.gif', format='GIF', append_images=imgs,
            save_all=True, duration=20, loop=0)

if __name__=='__main__':
    from config import config

    config.path = sys.argv[1]

    agent = MyAgent(config)
    print('AGENT INITIALIZED')
    
    test_class(agent)
