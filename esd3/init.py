# # -*- coding: utf-8 -*-
# """Inertia.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1rlvk1tm_0GRXyN3ITXKZXdktwqjJb0y2
# """

# # Commented out IPython magic to ensure Python compatibility.
# # %pip install ray
# # %pip install gymnasium

# """# Libraries"""

# import gymnasium as gym
# from gymnasium import spaces, vector
# import ray
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# import numpy as np
# import csv
# import random
# import logging

# """# Parameters"""

# CUSTOMERS = 3000
# PERIODS = 100
# STARTING_PRICES = [300,300,300]
# LARGEST_PRICE = 1000 # most you can decrease your prices in one period
# SMALLEST_PRICE = 0 # most you can increase your prices in one period


# def calculate_market_share(prices, inertia_factor=0.75, price_sensitivity_mean=0.2, price_sensitivity_std=0.05, smoothing_factor=0.05, total_customers=1000):
#     """
#     Calculate the market shares of firms based on prices, adjusting for customer inertia, smoothing,
#     and RL-ready optimization with variable price sensitivity modeled by a normal distribution.

#     Parameters:
#     prices (list): A list of prices for each firm.
#     inertia_factor (float): Proportion of customers resistant to switching (default 0.75).
#     price_sensitivity_mean (float): Mean sensitivity of customers to price differences.
#     price_sensitivity_std (float): Standard deviation for sensitivity to price differences.
#     smoothing_factor (float): A small factor to smooth price differences, preventing over-reactions to
#                               small price changes (default 0.05).
#     total_customers (int): The total number of customers in the market (default 1000).

#     Returns:
#     List: A list representing the number of customers each firm will have.
#     """

#     # Number of firms
#     num_firms = len(prices)

#     # Normalize prices: this ensures prices are comparable on a relative scale
#     min_price = np.min(prices)
#     max_price = np.max(prices)
#     normalized_prices = (prices - min_price) / (max_price - min_price + smoothing_factor)  # Add smoothing to avoid division by zero

#     # Initialize market shares
#     market_shares = np.zeros(num_firms)

#     # Generate price sensitivities for each customer based on a normal distribution
#     price_sensitivities = np.random.normal(price_sensitivity_mean, price_sensitivity_std, total_customers)
#     price_sensitivities = np.clip(price_sensitivities, 0, 1)  # Ensure sensitivities are within [0, 1]

#     # Calculate the relative attractiveness of each firm for each customer based on variable price sensitivity
#     for i in range(num_firms):
#         # Each customer has a different price sensitivity affecting their attractiveness to each firm
#         attractiveness = 1 - price_sensitivities * normalized_prices[i]

#         # Combine inertia and attractiveness
#         market_shares[i] = (1 - inertia_factor) * np.mean(attractiveness) + inertia_factor / num_firms

#     # Ensure the market shares sum to 1 by normalizing the shares
#     market_shares = market_shares / np.sum(market_shares)

#     # Scale market shares by the total number of customers
#     market_shares = market_shares * total_customers

#     # Apply final rounding for integer customer numbers
#     rounded_shares = np.round(market_shares).astype(int)

#     # Adjust rounding: ensure the total number of customers is preserved after rounding
#     difference = total_customers - np.sum(rounded_shares)
#     if difference != 0:
#         # Randomly adjust shares to account for rounding difference
#         indices = np.random.choice(num_firms, abs(difference), replace=False)
#         adjustment = 1 if difference > 0 else -1
#         for index in indices:
#             rounded_shares[index] += adjustment

#     market_shares = rounded_shares.tolist()
#     return market_shares


# print(calculate_market_share(STARTING_PRICES))

# class InertiaEnv(MultiAgentEnv):
#     def __init__(self, seed=None):
#         super(InertiaEnv, self).__init__()
#         #logging.basicConfig(level=logging.DEBUG)
#         self.t_steps = 0
#         self._num_agents = len(STARTING_PRICES)
#         self.agents = [f'agent_{i}' for i in range(self._num_agents)]

#         self.action_space = spaces.Dict({
#             agent: spaces.Box(low=SMALLEST_PRICE,high=LARGEST_PRICE,dtype=np.int32)
#             for agent in self.agents
#         })

#         self.observation_space = spaces.Dict({
#                 agent: spaces.Dict({'price': spaces.Box(low=SMALLEST_PRICE, high=LARGEST_PRICE,dtype=np.int32),
#                                     'market_prices': spaces.Box(low=0, high=LARGEST_PRICE,  shape=(len(STARTING_PRICES),), dtype=np.int32),
#                                     'market_quantities' :spaces.Box(low=0, high=np.inf,  shape=(len(STARTING_PRICES),) ,dtype=np.int32),
#                                              })
#                 for agent in self.agents
#             })
#         self.reset()

#     def step(self,actions):
#         self.t_steps += 1
#         self.current_period += 1
#         obs = {}
#         rewards = {}
#         terminateds = {}
#         truncateds = {}
#         info = {}
#         prices = [0 for i in range(len(actions))]
#         for i,(agent_id,action) in enumerate(actions.items()):
#             rewards[agent_id] = 0
#             truncateds[agent_id] = False
#             terminateds[agent_id] = False
#             prices[i] = action[0]

#         # This formula will be updated with the mathematical model generated from lit review
#         self.quantities = [i for i in calculate_market_share(prices)]
#         self.prices = prices

#         for i,(agent_id,action) in enumerate(actions.items()):
#             rewards[agent_id] += int(self.quantities[i]*prices[i])



#         truncateds['__all__'] = all(truncateds.values())

#         if self.current_period>PERIODS:
#             for agent_id, state in actions.items():
#                 terminateds[agent_id] = True
#             terminateds['__all__'] = all(terminateds.values())


#         for i,(agent_id, action) in enumerate(actions.items()):
#             obs[agent_id] = self._get_obs(i)
#         terminateds['__all__'] = all(terminateds.values())

#         return obs,rewards,terminateds,truncateds,info

#     def reset(self,*, seed=None, options=None):
#         self.current_period = 0
#         self.prices = STARTING_PRICES.copy()
#         self.quantities =calculate_market_share(STARTING_PRICES)
#         self.states = {
#             agent_id: {
#                 'price': STARTING_PRICES[i],
#                 'market_prices': np.array(STARTING_PRICES),
#                 'market_quantities': np.array(calculate_market_share(STARTING_PRICES))
#             }
#             for i,agent_id in enumerate(self.agents)
#         }
#         obs = {}
#         for i,agent_id in enumerate(self.agents):
#             obs[agent_id] = self._get_obs(i)

#         return obs, {}

#     def _get_obs(self,agent_id):
#         obs = {
#                 'price': np.array([self.prices[agent_id]],dtype=np.int32),
#                 'market_prices': np.array(self.prices,dtype=np.int32),
#                 'market_quantities': np.array(self.quantities,dtype=np.int32)
#             }
#         return obs

# env = InertiaEnv()
# obs, info =env.reset()
# obs

# for i in range(100):
#     obs,r,t,_,_ = env.step(env.action_space.sample())
#     if not env.observation_space.contains(obs):
#         print("uh oh",i)

# """# Training"""

# if ray.is_initialized():
#   ray.shutdown()
# ray.init(ignore_reinit_error=True)

# ray.available_resources()

# import os
# print(os.getcwd())
# save_dir = os.getcwd()

# from ray import air, tune
# from ray.rllib.algorithms.ppo import PPOConfig


# from ray.rllib.models import ModelCatalog
# from ray.rllib.policy.policy import PolicySpec
# from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.test_utils import check_learning_achieved
# from functools import partial

# num_policies = 3 # each agent will have its own policy
# timesteps_total = 1000 #1000000
# max_training_iteration = 10000
# num_agents = 3


# agent_ids = InertiaEnv().agents
# sym_policies = {agent_id: f"policy_agent_0" for agent_id in agent_ids}  # Symmetric
# asym_policies = {agent_id: f"policy_{agent_id}" for agent_id in agent_ids}  # Asymmetric
# def policy_mapping_fn(agent_id, episode, worker, *, policies=None, **kwargs):
#     return policies[agent_id]


# policies = asym_policies
# policy_mapping = partial(policy_mapping_fn, policies=policies)

# config = (
#         PPOConfig()
#         .environment(InertiaEnv)
#         .framework('torch')
#         .training(train_batch_size=200)
#         .api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)
#         .resources(num_gpus=0)
#         .debugging(seed=0)
#         #.rollouts(num_env_runners=1, num_envs_per_env_runner=1)
#         .multi_agent(policies=list(policies.values()),
#                      policy_mapping_fn=policy_mapping)
#     )
# config.num_env_runners=1
# config.num_sgd_iter = 10
# config.sgd_minibatch_size = 150
# config.entropy_coeff = 0.01

# stop = {
#         "timesteps_total": timesteps_total,
#         "training_iteration": max_training_iteration,
#     }


# # THE WAY OF CODING IS A BIT DIFFERENT HERE FROM THE SINGLE AGENT ONE.
# # WE CAN MAKE IT EXACTLY THE SAME IF IT DIDNT WORK. SPECIFICALLY, config() and results are defined a bit differently.
# results = tune.Tuner(
#         "PPO",
#         param_space=config.to_dict(),
#         run_config=air.RunConfig(
#             storage_path=save_dir,
#             stop=stop,
#             # Save a maximum X checkpoints at every N training_iteration
#             checkpoint_config=air.CheckpointConfig(
#                 checkpoint_frequency=5,
#                 checkpoint_at_end=True)
#         ),
#     ).fit()


# ###if args.as_test:
#  ###   check_learning_achieved(results, args.stop_reward)
# ray.shutdown()

# import os
# import ray
# import numpy as np
# from ray import tune
# from ray.tune.registry import get_trainable_cls
# from ray.rllib.algorithms.algorithm import Algorithm
# import matplotlib.pyplot as plt

# if ray.is_initialized():
#   ray.shutdown()
# ray.init(ignore_reinit_error=True)

# #Load a specific run vs. the one ran in the current session
# checkpoint_dir = "/content/PPO_2024-11-25_21-16-25/PPO_InertiaEnv_875dd_00000_0_2024-11-25_21-16-25/checkpoint_000000"
# algo = Algorithm.from_checkpoint(checkpoint_dir) # checkpoint

# from functools import partial

# MAX_PERIODS = 102  # max number of steps per episode
# num_simulations = 5  # Number of simulations to run

# agent_ids = InertiaEnv().agents
# sym_policies = {agent_id: f"policy_agent_0" for agent_id in agent_ids}  # Symmetric
# asym_policies = {agent_id: f"policy_{agent_id}" for agent_id in agent_ids}  # Asymmetric
# def policy_mapping_fn(agent_id, episode, worker, *, policies=None, **kwargs):
#     return policies[agent_id]


# policies = asym_policies
# policy_mapping = partial(policy_mapping_fn, policies=policies)


# env = InertiaEnv()
# data = []
# # for price in [j*5 for j in range(200)]:
# mean_reward = 0
# for i in range(num_simulations):
#     obs, info = env.reset()
#     terminated = {agent_id:False for agent_id in agent_ids}
#     terminated["__all__"] = False
#     steps = 0
#     while not terminated["__all__"]:
#         actions = {}
#         for agent in agent_ids:
#             a = algo.compute_single_action(
#                 observation=obs[agent],
#                 policy_id=policies[agent],
#             )
#             actions[agent] = a
#         # actions['agent_0'] = [price]
#         prev_obs = obs
#         obs, reward, terminated, truncated, info = env.step(actions)
#         steps += 1
#         data.append([obs['agent_0']['market_prices'],obs['agent_0']['market_quantities'],[r/1000 for r in reward.values()]])
       

# # Done: Prices all three firms, Customers (%) allocated by period, full profit data

# data
