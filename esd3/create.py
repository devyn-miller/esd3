# import os
# from itertools import product

# # Parameter lists
# inertia_factors = [0.65, 0.75, 0.85]
# price_sensitivity_means = [0.2, 0.3]
# price_sensitivity_stds = [0.01, 0.05]
# smoothing_factors = [0.01, 0.05]
# # checkpoints = [0, 5, 10]
# # inertia_factors = [0.65]
# # price_sensitivity_means = [0.2]
# # price_sensitivity_stds = [0.01]
# # smoothing_factors = [0.01]
# # checkpoints = [0]

# # Paste your code as a multi-line string below
# base_code = """

# import gymnasium as gym
# from gymnasium import spaces, vector
# import ray
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# import numpy as np
# import csv
# import random
# import logging



# CUSTOMERS = 3000
# PERIODS = 100
# STARTING_PRICES = [300,300,300]
# LARGEST_PRICE = 1000 # most you can decrease your prices in one period
# SMALLEST_PRICE = 0 # most you can increase your prices in one period


# def calculate_market_share(prices, inertia_factor={inertia_factor}, price_sensitivity_mean={price_sensitivity_mean}, price_sensitivity_std={price_sensitivity_std}, smoothing_factor={smoothing_factor}, total_customers=1000):

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
# print('hello')

# """

# # Generate all parameter combinations
# combinations = product(inertia_factors, price_sensitivity_means, price_sensitivity_stds, smoothing_factors)

# # Create directories and files
# for inertia_factor, price_sensitivity_mean, price_sensitivity_std, smoothing_factor in combinations:
#     # Create folder and file names
#     folder_name = f"inertia_{inertia_factor}_price_sensitivity_mean_{price_sensitivity_mean}_price_sensitivity_std_{price_sensitivity_std}_smooth_{smoothing_factor}"
#     file_name = f"{folder_name}.py"

#     # Create the folder
#     os.makedirs(folder_name, exist_ok=True)

#     # Write the file content
#     file_content = base_code.format(
#         inertia_factor=inertia_factor,
#         price_sensitivity_mean=price_sensitivity_mean,
#         price_sensitivity_std=price_sensitivity_std,
#         smoothing_factor=smoothing_factor
#     )
    
#     # Save the file inside the folder
#     with open(os.path.join(folder_name, file_name), "w") as f:
#         f.write(file_content)

# from ray import air, tune
# from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.models import ModelCatalog
# from ray.rllib.policy.policy import PolicySpec
# from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.test_utils import check_learning_achieved
# from gymnasium import spaces, vector
# import ray
# import numpy as np
# import gymnasium as gym
# import csv
# import random
# import logging