import os
import pickle
from functools import partial
from time import time
from typing import Any, Callable, List, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

import env_deepracer
from utils import gae_advantages

# percentage of allocated GPU memry
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.80'

ModuleDef = Any
Array = Any
Model = Any
PRNGKey = Any

class ActorCritic(nn.Module):
  """Class defining the actor-critic model."""

  num_outputs: int = 5

  @nn.compact
  def __call__(self, x):
    dtype = jnp.float32
    x = x.astype(dtype) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=512, name='hidden', dtype=dtype)(x)
    x = nn.relu(x)
    logits = nn.Dense(features=self.num_outputs, name='logits', dtype=dtype)(x)
    policy_log_probabilities = nn.log_softmax(logits)
    value = nn.Dense(features=1, name='value', dtype=dtype)(x)
    return policy_log_probabilities, value

class ACLight(nn.Module):

  num_outputs: int = 5

  @nn.compact
  def __call__(self, x):
    dtype = jnp.float32
    x = x.astype(dtype) / 255.
    x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv3',
                dtype=dtype)(x)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256, name='hidden', dtype=dtype)(x)
    x = nn.relu(x)
    logits = nn.Dense(features=self.num_outputs, name='logits', dtype=dtype)(x)
    policy_log_probabilities = nn.log_softmax(logits)
    value = nn.Dense(features=1, name='value', dtype=dtype)(x)
    return policy_log_probabilities, value

def create_train_state(params: flax.core.FrozenDict,
                        model: nn.Module, learning_rate: float, 
                        decaying_lr: bool = True, train_steps: int = 0) -> TrainState:
    """
    Creates train state from initialized 'params' and 'model'.
    set 'decaying_lr' to True to enable linear scheduler
    """
    if decaying_lr:
        lr = optax.linear_schedule(
            init_value = learning_rate, end_value=0.,
            transition_steps=train_steps)
    else:
        lr = learning_rate
    tx = optax.adam(lr)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx)
    return state

def collect_experience_jit(envs: env_deepracer.SubprocVecEnv, num_steps: int, state: TrainState)-> Tuple[Array, ...]:
    """
    collects experience.
    Envs - vectorized environments
    num_steps - steps to proceed in environments
    returns tuple of experience: (observations, taken_actions, rewards, values, action_log_probs, done_flags)
    """
    @jax.jit
    def apply_policy_one(obs: Array):
        log_prob, value = state.apply_fn({'params': state.params},
                                        obs)
        return log_prob, value

    observations_list = []
    actions_list = []
    rewards_list = []
    values_list = []
    log_probs_list = [] # log prob of taken action
    dones_list = []

    next_observations = envs.reset()
    for _ in range(num_steps+1):
        observations = next_observations
        log_probs, values = apply_policy_one(observations) 

        probs = np.exp(np.array(log_probs)) # n_envs X num_actions
        actions = []
        taken_log_probs = []
        for prob, lprob in zip(probs, log_probs):
            action = np.random.choice(probs.shape[1], p=prob)
            actions.append(action)
            taken_log_probs.append(lprob[action])

        next_observations, rewards, dones, info = envs.step(actions)

        observations_list.append(observations)
        actions_list.append(np.array(actions))
        rewards_list.append(rewards)
        values_list.append(values[..., 0])
        log_probs_list.append(np.array(taken_log_probs))
        dones_list.append(dones)
    experience =(
        jnp.stack(observations_list),
        jnp.stack(actions_list),
        jnp.stack(rewards_list),
        jnp.stack(values_list),
        jnp.stack(log_probs_list),
        jnp.stack(dones_list))
    return experience

def process_experience(
    experience: Tuple[Array, ...], 
    gamma: float = .99, 
    lambda_: float = .95
    ):
    """
    processes experience gained from 'collect_experience_jit'
    takes as input experience, gamma and lambda_
    gamma is a discount parameter and lambda_ is a parameter for GAE: generalized andvantage estimation.
    With lambda=0 we will ged Temporal Difference - like estimation and with lambda=1 we will get Mote-Carlo - like estimation.
    Helps to reduce bias at the cost of higher variance
    """
    states, actions, rewards, values, log_probs, dones = experience
    states = states[:-1]
    actions = actions[:-1] 
    rewards = rewards[:-1] 
    log_probs = log_probs[:-1]   
    dones = jnp.logical_not(dones[:-1]).astype(float) # REVERSED Done flags!!!!
    advantages = gae_advantages(rewards, dones, values, gamma, lambda_)
    returns = advantages + values[:-1]
    trajectories = (states, actions, log_probs, returns, advantages)
    num_agents, actor_steps = states.shape[:2]
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(map(
      lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories))
    return trajectories

def loss_fn(
        params: flax.core.FrozenDict,
        apply_fn: Callable[..., Any],
        minibatch: Tuple,
        clip_param: float,
        vf_coeff: float,
        entropy_coeff: float):
    """
    PPO loss function.
    clip_param - param for clipping in PPO loss
    vf_coeff - coefficient near value function loss
    entropy_coeff - coefficient of the term responsible for entropy. Higher the term => lower model's confidence. It helps exploration
    
    mimibatch: (
        states: (batch_size, h, w, c*num_framestack)
        actions: (batch_size, )
        old_log_probs: (batch_size, )
        returns: (batch_size, )
        advantages: (batch_size, )
    )
                
    """
    states, actions, old_log_probs, returns, advantages = minibatch
    log_probs, values  = apply_fn({'params': params}, states,)
    values = values[:, 0]  # Convert shapes: (batch, 1) to (batch, ).
    probs = jnp.exp(log_probs)

    value_loss = jnp.mean(jnp.square(returns - values), axis=0)

    entropy = jnp.sum(-probs*log_probs, axis=1).mean()
    # taking only log_probs of already taken actions with policy before updates
    log_probs_act_taken = jax.vmap(lambda lp, a: lp[a])(log_probs, actions)
    # ratios of old and current probabilities for further clipping
    ratios = jnp.exp(log_probs_act_taken - old_log_probs)
    # Advantage normalization (following the OpenAI baselines).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = ratios * advantages
    clipped_loss = advantages * jax.lax.clamp(1. - clip_param, ratios,
                                                1. + clip_param)
    ppo_loss = -jnp.mean(jnp.minimum(pg_loss, clipped_loss), axis=0)

    return ppo_loss + vf_coeff*value_loss - entropy_coeff*entropy

@partial(jax.jit, static_argnums=(2,))
def train_step(
        state: TrainState,
        trajectories: Tuple,
        batch_size: int,
        *,
        clip_param: float,
        vf_coeff: float,
        entropy_coeff: float):
    """
    Train step function. Takes processed experience, trajectories and
    batch_size
    clip_param - param for clipping in PPO loss
    vf_coeff - coefficient near value function loss
    entropy_coeff - coefficient of the term responsible for entropy. Higher the term => lower model's confidence. It helps exploration
    """
    # total number of batches i.e. iterations in out experience
    iterations = trajectories[0].shape[0] // batch_size
    # reshape arrays to (iterations, batch_size, ...)
    trajectories = jax.tree_map(
        lambda x: x.reshape((iterations, batch_size) + x.shape[1:]), trajectories)
    loss = 0.
    #training loop
    for batch in zip(*trajectories):
        grad_fn = jax.value_and_grad(loss_fn)
        l, grads = grad_fn(state.params, state.apply_fn, batch, clip_param, vf_coeff,
                        entropy_coeff)
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss

def get_initial_params(key: PRNGKey, model: nn.Module, obs_shape) -> flax.core.FrozenDict:
    """
    init params of model using PRNGKey and obs_shape.
    for initialization, a batch is created with shaoe (1, obs_shape) and then forward passed through the model
    return frozen dict of parameters
    """
    obs_batch = np.ones((1,)+obs_shape)
    variables = model.init(key, obs_batch)
    params = variables['params']
    return params

def policy_test(envs: env_deepracer.SubprocVecEnv, state: TrainState, n_episodes: int = 3, n_actors: int = 2) -> float:
    """
    applying greedy policy with current advantage estimator in 'state' for n_episodes*n_actors episodes in total
    return mean score
    """
    @jax.jit
    def apply_policy_one(obs: Array):
        # apply-model-once function for jitting 
        log_prob, value = state.apply_fn({'params': state.params,}, obs)
        return log_prob, value

    total_reward = 0.0
    for _ in range(n_episodes):
        obs = envs.reset()
        done_array = jnp.zeros(n_actors)
        # loop below collects rewards and breaks only when all environments in subprocesses raised a Done flag. 
        while not done_array.all():
            log_probs, _ = apply_policy_one(obs)
            actions = np.argmax(log_probs, axis=-1).tolist()
            obs, reward, done, _ = envs.step(actions)
            done_array += done
            # when Done flag is raised in one environment its rewards does not counts
            total_reward += (reward @ jnp.logical_not(done_array)).item()

    total_epizodes = n_actors * n_episodes
    return total_reward / total_epizodes

def save_state(dir: str, filename: str, state: TrainState):
    # pickling state dictionary. State has all information about current train state: weights, optimizer stats and current step
    state_dict = flax.serialization.to_state_dict(state)
    if dir:
        with open(os.path.join(dir,filename + '.pickle'), 'wb') as handle:
            pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_state(path: str, state: TrainState, lin_scheduler: bool) -> TrainState:
    """
    loading state from pickled dictionary. In python 3.7 (and below?) use pickle5 library
    """
    from flax.core import freeze

    with open(path, 'rb') as handle:
        state_dict = pickle.load(handle)
    params = freeze(jax.tree_map(lambda x: jnp.array(x), state_dict['params']))
    step = state_dict['step']

    # function that restores all objects in state.opt_state
    def restore_opt_state(opt_state_dict):
        """
        Information about optimizer consists of state of optimizer (we have Adam) and state of scheduler (wa have linear or None)
        We are importing classes for Adam and scheduler below and filling them with values from loaded state dict.
        """
        from copy import deepcopy

        from optax._src.base import EmptyState
        from optax._src.transform import ScaleByAdamState, ScaleByScheduleState
        opt_state_dict = deepcopy(opt_state_dict)
        opt_state_dict = jax.tree_map(lambda x: jnp.array(x), opt_state_dict)
        state1 = ScaleByAdamState(
            count=opt_state_dict['0']['count'], 
            mu=freeze(opt_state_dict['0']['mu']),
            nu=freeze(opt_state_dict['0']['nu'])
        )
        # scheduler information is just a NamedTuple
        if lin_scheduler:
            state2 = ScaleByScheduleState(count=jnp.array(state_dict['step']))
        else:
            state2 = EmptyState()

        return [state1, state2]
    opt_state = restore_opt_state(state_dict['opt_state'])
    
    return state.replace(params=params, step=step, opt_state=opt_state)

def train(config):
    # config class to dict
    config_dict = dict(vars(config))
    for key in list(config_dict.keys()):
        if '__' in key:
            del config_dict[key]

    key = jax.random.PRNGKey(0)

    #initing midel class from model dict for logging convenience
    model = MODEL_DICT[config.backbone]()
    params = get_initial_params(key, model, obs_shape=config.obs_shape)

    # num of updates for one loop step (experience collection)
    iterations_per_step = len(config.ports)*config.actor_steps// config.batch_size
    # number of experience_collection -> update_state iterations
    loop_steps = config.total_frames // (config.actor_steps*len(config.ports))

    state = create_train_state(params=params, model=model, learning_rate=config.learning_rate, 
                        decaying_lr=config.decaying_lr_and_clip_param, train_steps=loop_steps * config.num_epochs * iterations_per_step)
    # params already saved in state
    del params

    # making vectorized environment
    env_funs = [env_deepracer.make_env_fn(port=port, num_stack=config.num_stack, num_skip=config.num_skip)  for port in config.ports]
    envs = env_deepracer.SubprocVecEnv(env_funs, image_shape=config.image_shape, stacks=config.num_stack)
    envs.reset()
    print('ENVIRONMENTS CREATED')
    if config.load_state:
        state = load_state(path=config.load_state, state=state, lin_scheduler=config.decaying_lr_and_clip_param)
        print(f'State {config.load_state} loaded with step {state.step}')

    if config.wandb:
        wandb.init(project='deepracer', config=config_dict)
        wandb.run.name += '_' + config.run_name
    #start_step is not 0 when we have loaded some pretrained weights into our state
    start_step = int(state.step) // config.num_epochs // iterations_per_step

    print(f'Starting step: {start_step}')
    best_score = 0
    for step in range(start_step, loop_steps):
        step_start = time()
        print(f'STEP {step}/{loop_steps}')
        if step % config.test_frequency == 0:
            print('testing...')
            test_out = policy_test(envs, state, config.runs_in_test, n_actors=len(config.ports))
            if config.wandb:
                wandb.log({'RunScore': test_out})
            print(f'test ended with {test_out:.1f} score')
            if test_out > best_score:
                if config.save:
                    #saving the best state
                    save_state(config.model_dir, f'best_{int(test_out)}_' + wandb.run.name , state)
                best_score = test_out
        # coefficient for linear decaying of clipping parameter
        alpha = 1. - step/loop_steps if config.decaying_lr_and_clip_param else 1.
        print('collecting experience....')
        all_experience = collect_experience_jit(envs, num_steps=config.actor_steps, state=state)
        trajectories = process_experience(all_experience, config.gamma, config.lambda_)
        clip_param = config.clip_param * alpha
        
        # update state iterations
        for _ in range(config.num_epochs):
            assert len(trajectories[0]) == config.actor_steps*len(config.ports)
            permutation = np.random.permutation(len(config.ports) * config.actor_steps)
            trajectories = tuple(x[permutation] for x in trajectories)
            state, loss = train_step(
                state, trajectories, config.batch_size,
                clip_param=clip_param,
                vf_coeff=config.vf_coeff,
                entropy_coeff=config.entropy_coeff)
            if config.wandb:
                wandb.log({'step_loss': loss.item()})
        step_time = int(time() - step_start)
        print(f'STEP TIME {step_time}')

        if (step) % config.checkpoint_frequency == 0:
            if config.save:
                save_state(config.model_dir, wandb.run.name+'_'+str(step), state)
    if config.save:
        save_state(config.model_dir, wandb.run.name+'_'+str(step), state)

    return state

MODEL_DICT = {
    'SimpleConv': ActorCritic,
    'SCLight': ACLight,
}

def test_collect_experience_jit(config):
    print('-------------------EXPERIENCE COLLECTION TEST---------------------')
    loop_steps = config.total_frames // (config.actor_steps)
    key = jax.random.PRNGKey(0)

    model = MODEL_DICT[config.backbone]()
    params = get_initial_params(key, model, obs_shape=config.obs_shape)

    iterations_per_step = config.actor_steps// config.batch_size
    loop_steps = config.total_frames // config.actor_steps

    state = create_train_state(params=params, model=model, learning_rate=config.learning_rate, 
                        decaying_lr=config.decaying_lr_and_clip_param, train_steps=loop_steps * config.num_epochs * iterations_per_step)

    del params

    env_funs = [env_deepracer.make_env_fn(port=port, num_stack=config.num_stack, num_skip=config.num_skip)  for port in config.ports]
    envs = env_deepracer.SubprocVecEnv(env_funs, image_shape=config.image_shape, stacks=config.num_stack)
    envs.reset()
    print('ENVIRONMENTS CREATED')

    all_experience = collect_experience_jit(envs, num_steps=config.actor_steps, state=state)

    observations, actions, rewards, values, log_probs, dones = all_experience

    print('-------------------observations----------------------------')
    print(observations.shape)
    print((config.actor_steps, len(config.ports)) + config.image_shape + (config.num_stack,))
    print('-------------------actions----------------------------')
    print(actions.shape)
    print((config.actor_steps, len(config.ports)))
    print('-------------------rewards----------------------------')
    print(rewards.shape)
    print((config.actor_steps, len(config.ports)))
    print('-------------------values----------------------------')
    print(values.shape)
    print((config.actor_steps, len(config.ports)))
    print('-------------------log_probs----------------------------')
    print(log_probs.shape)
    print((config.actor_steps, len(config.ports)))
    print('-------------------dones----------------------------')
    print(dones.shape)
    print((config.actor_steps, len(config.ports)))
    print('-------------------processing experience----------------')
    trajectories = process_experience(all_experience)
    states, actions, log_probs, returns, advantages = trajectories
    for x in trajectories:
        print(x.shape)

def test_policy_test(config):
    print('-------------------POLICY TESTING FUNCTION---------------------')
    loop_steps = config.total_frames // (config.actor_steps)
    key = jax.random.PRNGKey(0)

    model = MODEL_DICT[config.backbone]()
    params = get_initial_params(key, model, obs_shape=config.obs_shape)

    iterations_per_step = config.actor_steps// config.batch_size
    loop_steps = config.total_frames // config.actor_steps

    state = create_train_state(params=params, model=model, learning_rate=config.learning_rate, 
                        decaying_lr=config.decaying_lr_and_clip_param, train_steps=loop_steps * config.num_epochs * iterations_per_step)

    del params

    if config.load_state:
        state = load_state(path=config.load_state, state=state, lin_scheduler=config.decaying_lr_and_clip_param)
        print(f'State {config.load_state} loaded with step {state.step}')

    env_funs = [env_deepracer.make_env_fn(port=port, num_stack=config.num_stack, num_skip=config.num_skip)  for port in config.ports]
    envs = env_deepracer.SubprocVecEnv(env_funs, image_shape=config.image_shape, stacks=config.num_stack)
    envs.reset()
    print('ENVIRONMENTS CREATED')

    print(
        'MEAN REWARD: ',
        policy_test(envs=envs, state=state, n_episodes=2, n_actors=len(config.ports))
    )

if __name__=='__main__':

    class config:
        ports = [8887, 8886, 8885, 8884, 8883, 8882]
        runs_in_test = 2
        num_stack = 4
        num_skip = 1
        image_shape = (32, 32)
        total_frames = 6_000_000
        actor_steps = 256
        batch_size = 256
        learning_rate = 2.5e-4

        backbone = 'SCLight'

        decaying_lr_and_clip_param = True
        gamma = .99
        lambda_ = .95
        clip_param = .1
        num_epochs = 3
        vf_coeff = 0.5
        entropy_coeff = 0.01

        checkpoint_frequency = 50
        test_frequency = 4
        
        run_name = f'{len(ports)}env_{backbone}_{num_stack}st{num_skip}sk'
        model_dir = './save/'

        load_state = None
        save = True
        wandb = True

        obs_shape = (64, 64, 4)

    print('START') 
    train(config)
