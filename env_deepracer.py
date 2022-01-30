import functools
import multiprocessing as mp

import gym
import jax
import jax.numpy as jnp
import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

import env_utils

m.patch()


class DeepracerZMQClient:
    def __init__(self, host="127.0.0.1", port=8888):
        self.host = host
        self.port = port
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.set(zmq.SNDTIMEO, 200000)
        self.socket.set(zmq.RCVTIMEO, 200000)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
    
    def set_agent_ready(self):
        packed_msg = msgpack.packb({"Agent Ready": 1})
        self.socket.send(packed_msg)

    def recieve_response(self):
        packed_response = self.socket.recv()
        response = msgpack.unpackb(packed_response)
        return response

    def send_msg(self, msg: dict):
        packed_msg = msgpack.packb(msg)
        self.socket.send(packed_msg)

        response = self.recieve_response()
        return response

class DeepracerEnvHelper:
    def __init__(self, port=8888):
        self.zmq_client = DeepracerZMQClient(port=port)
        self.zmq_client.set_agent_ready()
        self.obs = None
        self.previous_done = False

    def send_act_rcv_obs(self, action):
        action_dict = {"action": action}
        self.obs = self.zmq_client.send_msg(action_dict)
        self.previous_done = self.obs['_game_over']
        return self.obs
    
    def env_reset(self):
        if self.obs is None: # First communication to zmq server
            self.obs = self.zmq_client.recieve_response()
        elif self.previous_done: # To prevent dummy episode on already done env
            pass
        else: # Can't reset env before episode completes - Passing '1' until episode completes
            action = 1
            done = False
            while not done:
                self.obs = self.send_act_rcv_obs(action)
                done = self.obs['_game_over']
            self.previous_done = True

        return self.obs
    
    def unpack_rl_coach_obs(self, rl_coach_obs):
        observation = rl_coach_obs['_next_state']
        reward = rl_coach_obs['_reward']
        done = rl_coach_obs['_game_over']
        info = rl_coach_obs['info']
        if type(info) is not dict:
            info = {}
        info['goal'] = rl_coach_obs['_goal']
        return observation, reward, done, info

class DeepracerGymEnv(gym.Env):
    def __init__(self, port=8888):
        self.action_space = gym.spaces.Discrete(5)
        self.deepracer_helper = DeepracerEnvHelper(port=port)
    
    def reset(self):
        observation = self.deepracer_helper.env_reset()
        return observation
    
    def step(self, action):
        rl_coach_obs = self.deepracer_helper.send_act_rcv_obs(action)
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(rl_coach_obs)
        return observation, reward, done, info

def _worker(remote, parent_remote, env_fn) -> None:

    parent_remote.close()
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    info["terminal_observation"] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            print('PROCESS ENDED')
            break

class SubprocVecEnv:

    def __init__(self, env_fns, start_method=None, image_shape=(64,64), stacks=4):
        out_shape = image_shape + (stacks,)
        self.resize_fun = jax.vmap(functools.partial(resize, shape=out_shape), in_axes=0, out_axes=0)

        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step_async(self, actions) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return self.resize_fun(_flatten_obs(obs)), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return self.resize_fun(_flatten_obs(obs))

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

def resize(image, shape):
    h, w, c = shape
    return jax.image.resize(image, shape=shape, method='bilinear')[h//2:, :, :]

def _flatten_obs(obs):
    stacked = jnp.stack(obs)
    return stacked

def create_env(port=8888, num_stack: int = 4, num_skip: int = 4):

    env = DeepracerGymEnv(port=port)
    env_wrapped = env_utils.BasicWrapperOneChannel(env)
    env_wrapped = env_utils.FrameSkip(env_wrapped, num_skip=num_skip)
    env_wrapped = env_utils.FrameStackB(env_wrapped, num_stack=num_stack)
    env_wrapped.reset()
    print(f"Deepracer Environment on port {port} connected succesfully")
    return env_wrapped

def make_env_fn(port, num_stack=4, num_skip=4):
    env_fn = functools.partial(create_env, port=port, num_stack=num_stack, num_skip=num_skip)
    return env_fn

def test_two_ports_consequentially():
    env = DeepracerGymEnv(port=8887)
    obs = env.reset()
    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    from time import time
    st = time()
    for _ in range(500):
        observation, reward, done, info = env.step(np.random.randint(5))
        steps_completed += 1 
        total_reward += reward
        if done:
            episodes_completed += 1
            print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
            print(f"mean time {(time() - st)/steps_completed:.6f}")
            st = time()
            steps_completed = 0
            total_reward = 0

    env = DeepracerGymEnv(port=8886)
    obs = env.reset()
    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    st = time()
    for _ in range(500):
        observation, reward, done, info = env.step(np.random.randint(5))
        steps_completed += 1 
        total_reward += reward
        if done:
            episodes_completed += 1
            print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
            print(f"mean time {(time() - st)/steps_completed:.6f}")
            st = time()
            steps_completed = 0
            total_reward = 0

def test_maker_function():
    env_fn = make_env_fn(port=8886)
    env = env_fn()#create_env(port=8887)  #env_fn()
    obs = env.reset()
    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    from time import time
    st = time()
    for _ in range(500):
        observation, reward, done, info = env.step(np.random.randint(5))
        steps_completed += 1 
        total_reward += reward
        if done:
            episodes_completed += 1
            print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
            print(f"mean time {(time() - st)/steps_completed:.6f}")
            st = time()
            steps_completed = 0
            total_reward = 0

    print(observation.shape)
    print(observation)

def test_vec_env():
    env_func_list = [make_env_fn(port=8887), make_env_fn(port=8886)]
    envs = SubprocVecEnv(env_func_list, image_shape=(32,64), stacks=4)
    print('CREATED')
    envs.reset()
    observation, reward, done, info = envs.step(np.array([2,2]))
    print(reward)
    print(info)
    print(done)
    print(observation.shape)

    from time import time

    st = time()
    for _ in range(50):
        observation, reward, done, info = envs.step(np.array([2,2]))

    print('STEPS COMPLETED!')
    print(f'Mean step time is {(time()-st)/50:4f} sec')

if __name__ == '__main__':
    #test_vec_env()
    #test_two_ports_consequentially()
    #test_maker_function()
    pass
