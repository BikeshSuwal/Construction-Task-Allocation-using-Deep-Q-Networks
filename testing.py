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
import matplotlib.pyplot as plt

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

# Split data (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert states to feature vectors
def extract_state(row):
    return torch.tensor(np.array(row[feature_columns].values, dtype=np.float32), dtype=torch.float32)


# Define input dimension
input_dim = len(feature_columns)

# Define output dimension as the number of possible worker-task pairs
workers = data['worker_id'].unique()
tasks = data['task_id'].unique()
num_workers = len(workers)
num_tasks = len(tasks)
output_dim = num_workers * num_tasks

# Function to decode action index back to (worker_id, task_id)
def decode_action(action_idx, workers, tasks):
    worker_idx = action_idx // num_tasks
    task_idx = action_idx % num_tasks
    return workers[worker_idx], tasks[task_idx]

# Define DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Model hyperparameters
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
alpha = 0.001  # Learning rate
batch_size = 64
memory_size = 10000
num_episodes = 1000

# Initialize DQN and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(input_dim, output_dim).to(device)
model.load_state_dict(torch.load("trained_dqn_model.pth", map_location=device))
model.eval()  # Set to evaluation mode
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

#--------------------------------

# To track mispredictions
mispredictions = []

# Testing the model
correct_predictions = 0
total_predictions = len(test_data)
all_rewards = []

# Testing loop
for _, row in test_data.iterrows():
    # Convert current state to tensor
    state = extract_state(row).to(device)

    # Get model's prediction
    with torch.no_grad():
        predicted_action_idx = model(state).argmax().item()

    # Decode the predicted action into worker and task IDs
    predicted_worker, predicted_task = decode_action(predicted_action_idx, workers, tasks)

    # Get actual worker and task IDs from the test data
    actual_worker_id = row['worker_id']
    actual_task_id = row['task_id']

    # Get the full data rows for both predicted and actual assignments
    predicted_assignment = test_data[
        (test_data['worker_id'] == predicted_worker) &
        (test_data['task_id'] == predicted_task)
    ]



    actual_assignment = test_data[
        (test_data['worker_id'] == actual_worker_id) &
        (test_data['task_id'] == actual_task_id)
    ]


    # Calculate rewards only if we have valid assignments
    if not predicted_assignment.empty and not actual_assignment.empty:
        predicted_reward = calculate_reward(predicted_assignment.iloc[0])
        actual_reward = calculate_reward(actual_assignment.iloc[0])

        reward_diff = abs(predicted_reward - actual_reward)
        all_rewards.append(reward_diff)

        # Check if within acceptable deviation
        if reward_diff < 1.0:
            correct_predictions += 1
        else:
            mispredictions.append({
                'Predicted Worker': predicted_worker,
                'Predicted Task': predicted_task,
                'Actual Worker': actual_worker_id,
                'Actual Task': actual_task_id,
                'Reward Difference': reward_diff
            })
    else:
        mispredictions.append({
            'Predicted Worker': predicted_worker,
            'Predicted Task': predicted_task,
            'Actual Worker': actual_worker_id,
            'Actual Task': actual_task_id,
            'Reward Difference': "Invalid Assignment"
        })

# **Summary Statistics**
accuracy = correct_predictions / total_predictions * 100
average_reward_difference = np.mean(all_rewards) if all_rewards else float('nan')
median_reward_difference = np.median(all_rewards) if all_rewards else float('nan')

# Print summary
print(f"\n=== Model Evaluation Summary ===")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Average Reward Difference: {average_reward_difference:.2f}")
print(f"Median Reward Difference: {median_reward_difference:.2f}")

# **Display Mispredictions**
if mispredictions:
    mispredictions_df = pd.DataFrame(mispredictions)
    print("\n=== Mispredictions Summary ===")
    print(mispredictions_df)

# **Plot Reward Differences**
plt.figure(figsize=(8, 5))
plt.hist(all_rewards, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Reward Difference")
plt.ylabel("Frequency")
plt.title("Distribution of Reward Differences")
plt.grid()
plt.show()

