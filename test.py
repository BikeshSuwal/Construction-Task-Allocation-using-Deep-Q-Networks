import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from collections import deque
import gym
from gym import spaces
from tqdm import tqdm

# Load dataset
file_path = "updated_construction_dataset.csv"
try:
    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError("Empty dataset")
except Exception as e:
    raise Exception(f"Failed to load dataset: {e}")

# Define reward function
def calculate_reward(row):
    # Check if row contains NaN values or is empty
    if row.isnull().any():
        print(f"Warning: Missing values detected in row. Assigning penalty.")
        return -5  # Assign a penalty for missing data

    reward = 0
    reward += 1 if row.get('completed_on_time', 0) == 1 else -1
    reward += row.get('success_score', 0)
    reward += row.get('quality_score', 0)
    reward -= abs(1 - row.get('skill_match_score', 1)) * 0.5  # Default skill_match_score to 1 if missing
    reward += row.get('experience_adjusted_reliability', 0) * 0.2

    return reward


data['reward'] = data.apply(calculate_reward, axis=1)

# Feature columns (already preprocessed)
# Exclude 'worker_id' and 'task_id' from feature columns
feature_columns = [col for col in data.columns if col not in ['worker_id', 'task_id']]

# print(feature_columns)

#-------------------------------

# import streamlit as st
# import torch
# import numpy as np
# import pandas as pd

# Load the trained model
class DQN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load dataset to extract workers and tasks
dataset_path = "updated_construction_dataset.csv"
data = pd.read_csv(dataset_path)

workers = data["worker_id"].unique()
tasks = data["task_id"].unique()
num_workers = len(workers)
num_tasks = len(tasks)

# Get current dataset columns
current_columns = list(data.columns)

# Find missing columns
missing_columns = [col for col in feature_columns if col not in current_columns]
extra_columns = [col for col in current_columns if col not in feature_columns]

# Print results
print(f"Feature Columns in Model: {feature_columns}")
print(len(feature_columns))
print(f"Current Data Columns: {current_columns}")
print(len(current_columns))
print(f"Missing Columns: {missing_columns}")
print(len(missing_columns))
print(f"Unexpected Extra Columns: {extra_columns}")
print(len(extra_columns))
