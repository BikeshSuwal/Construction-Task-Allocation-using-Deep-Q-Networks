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
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# Experience replay memory
memory = deque(maxlen=10000) # Limit memory size to avoid slowdowns

#---------------------------------------------
#Defining the custom environment

class ResourceAllocationEnv(gym.Env):
    def __init__(self, data):
        super(ResourceAllocationEnv, self).__init__()

        # Define action and observation space
        self.workers = data['worker_id'].unique()
        self.tasks = data['task_id'].unique()
        self.num_workers = len(self.workers)
        self.num_tasks = len(self.tasks)
        self.action_space = spaces.Discrete(self.num_workers * self.num_tasks)  # Worker-task pairs
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(feature_columns),), dtype=np.float32)

        # Initialize data
        self.data = data
        self.reset()

    def reset(self):
        # Reset the environment to initial state
        self.worker_availability = {worker: True for worker in self.workers}  # All workers available
        self.task_status = {task: False for task in self.tasks}  # All tasks unassigned
        self.current_state = self._get_initial_state()
        return self.current_state

    def _get_initial_state(self):
        # Return the initial state (all workers available, all tasks unassigned)
        return np.zeros(len(feature_columns), dtype=np.float32)

    def step(self, action):
        worker_idx = action // self.num_tasks
        task_idx = action % self.num_tasks
        worker = self.workers[worker_idx]
        task = self.tasks[task_idx]

        # If invalid action (worker busy or task assigned)
        if not self.worker_availability[worker] or self.task_status[task]:
            reward = -10  # High penalty for invalid action
            next_state = self.current_state  # No state change
            done = False
            return next_state, reward, done, {}

        # Assign worker to task
        self.worker_availability[worker] = False  # Mark worker as busy
        self.task_status[task] = True  # Mark task as assigned

        # Update state
        next_state = self._update_state(worker, task)

        # Calculate reward
        reward = self._calculate_reward(worker, task)

        # Check if all tasks are assigned (end condition)
        done = all(self.task_status.values())

        return next_state, reward, done, {}


    def _update_state(self, worker, task):
        state = np.zeros(len(feature_columns), dtype=np.float32)
        
        # Get worker-task row data
        row = self.data[(self.data['worker_id'] == worker) & (self.data['task_id'] == task)]
        
        if not row.empty:
            state = row[feature_columns].values[0]  # Extract feature values
        else:
            print(f"Warning: No data found for worker {worker} and task {task}. Using default state.")
            state = self.data[feature_columns].mean().values  # Use dataset mean instead of zero vector

        return state


    def _calculate_reward(self, worker, task):
        # Get the row corresponding to (worker, task)
        row = self.data[(self.data['worker_id'] == worker) & (self.data['task_id'] == task)]
        
        if row.empty:  # Handle missing data case
            print(f"Warning: No data found for worker {worker} and task {task}. Assigning penalty.")
            return -10  # Assign a penalty for missing assignments
        
        return calculate_reward(row.iloc[0])  # Safe access now

# Training loop
# Initialize the environment
env = ResourceAllocationEnv(train_data)

# Training loop
for episode in tqdm(range(num_episodes), desc="Episodes", ncols=100):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Random action (Explore)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                action = model(state_tensor).argmax().item()  # Best action (Exploit)

        # Take action and observe next state, reward, and done
        next_state, reward, done, _ = env.step(action)

        # Store experience in memory
        memory.append((state, action, reward, next_state, done))

        # Train model using experience replay
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
            
            # Compute Q-values
            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = model(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

            # Compute loss and update model
            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        episode_reward += reward

    # Decay epsilon over time
    epsilon = max(0.01, epsilon * 0.995)


torch.save(model.state_dict(), "trained_dqn_model.pth")