import os
import subprocess
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

# Base directory
base_dir = "/Users/devynmiller/Downloads/esd3out/esd3/"
# Output log file
output_log_file = "all_outputs.txt"

# Initialize the log file
with open(output_log_file, "w") as log_file:
    log_file.write("Execution Log for All Python Files\n")
    log_file.write("=" * 50 + "\n\n")

# Traverse the directory tree
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".py"):  # Only look for Python files
            file_path = os.path.join(root, file)
            print(f"Running {file_path}...")

            # Append output to the log file
            with open(output_log_file, "a") as log_file:
                log_file.write(f"Running file: {file_path}\n")
                log_file.write("-" * 50 + "\n")

                try:
                    # Run the Python file and capture output
                    result = subprocess.run(
                        ["python", file_path],
                        capture_output=True,
                        text=True,
                        check=True  # Raise an exception if the script fails
                    )
                    
                    # Write the successful output
                    log_file.write("Output:\n")
                    log_file.write(result.stdout)
                    log_file.write("\n")
                    log_file.write("-" * 50 + "\n")
                    print(f"Successfully ran {file_path}")

                except subprocess.CalledProcessError as e:
                    # Write error information to the log
                    log_file.write("Error Output:\n")
                    log_file.write(e.stderr)
                    log_file.write("\n")
                    log_file.write("-" * 50 + "\n")
                    print(f"Error while running {file_path}")
