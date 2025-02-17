# Construction Task Allocation using Deep Q-Networks (DQN)

## Overview

This project aims to optimize task allocation in construction management using a Deep Q-Network (DQN) in a custom Gym environment. Initially, a Q-table was used, but due to large state space issues, a DQN model was implemented. The model is trained using synthetic data and incorporates various features to improve decision-making.

## Dataset Generation

A dataset was created using the `ConstructionLaborDataGenerator.py` script, containing:

- **20 workers** and **100 tasks**
- **500 assignments** between them

## Data Preprocessing

To ensure consistency and effective training, the dataset was preprocessed with the following steps:

### 1. **Categorical Encoding**

- One-hot encoding applied to categorical features:
  - `primary_skill`
  - `task_type`
  - `completed_on_time`

### 2. **Normalization**

- Applied MinMaxScaler to numerical features:
  - `estimated_duration_x`
  - `actual_duration`
  - `quality_score`
  - `success_score`
  - `estimated_duration_y`
  - `years_experience`

### 3. **Feature Engineering**

- **Skill Match Score:** Measures the match between worker skill and task requirement.

  \(skill\_match\_score = \frac{skill\_level}{required\_skill\_level}\)

- **Experience-adjusted reliability:** Incorporates reliability, experience, and task completion rate.

  \(experience\_adjusted\_reliability = reliability\_score \times \log(1 + years\_experience) \times avg\_task\_completion\_rate\)

  Normalized to ensure consistency across reward components.

## Problem Definition

- **State Space:** Worker-task feature vector
- **Action Space:** Assigning a worker to a task (initially 20×100=2000, later reduced to 10×10=100)
- **Reward Function:** Based on task completion, skill match, and reliability.

## Model Implementation

- **Initial Approach:** Q-table (failed due to large state space)
- **Final Approach:** DQN model implemented using PyTorch

### DQN Architecture

- **Input:** Feature vector of worker-task pair
- **Hidden Layers:**
  - Fully connected layer (64 neurons, ReLU activation)
  - Fully connected layer (64 neurons, ReLU activation)
- **Output:** Q-values for each action
- **Loss Function:** MSELoss
- **Optimizer:** Adam

### Experience Replay & Training

- **Memory buffer:** deque (max length: 10,000)
- **Exploration strategy:** ε-greedy (ε decays over episodes)
- **Training episodes:** 1000
- **Batch size:** 64
- **Discount factor (γ):** 0.9
- **Learning rate (α):** 0.001

## Custom Gym Environment

A **ResourceAllocationEnv** was created to simulate task assignments.

- **Observation Space:** Feature vector (state representation)
- **Action Space:** Worker-task pair selection
- **Reward Mechanism:** Encourages successful, timely, and skill-matched assignments

## Key Challenges & Solutions

1. **Q-table limitations** → Switched to DQN
2. **Long training time** → Reduced action space & optimized memory buffer
3. **Synthetic data mismatch with real-world scenarios** → Need real-world data for better performance

## Usage Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data generation (if needed):
   ```bash
   python ConstructionLaborDataGenerator.py
   ```
3. Train the model:
   ```bash
   python train_dqn.py
   ```
4. Test the trained model:
   ```bash
   python test_model.py
   ```

## Debugging Column Mismatches

To identify missing columns:

```python
missing_columns = [col for col in feature_columns if col not in data.columns]
print("Missing columns:", missing_columns)
```

If columns are missing:

- Check preprocessing steps for accidental drops.
- Verify column names match expected feature set.
- Inspect NaN values that may have caused removal.

## Future Improvements

- **Integrate real-world construction data**
- **Implement multi-agent reinforcement learning**
- **Optimize hyperparameters using Bayesian search**
- **Develop a visualization dashboard for results**



