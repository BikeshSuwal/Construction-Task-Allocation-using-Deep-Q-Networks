import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
file_path = "full_construction_dataset.csv"  # Replace with the correct path to your file
data = pd.read_csv(file_path)

# Encoding 'primary_skill' and 'task_type' using one-hot encoding
data = pd.get_dummies(data, columns=['primary_skill', 'task_type'], drop_first=True)

# Encoding 'completed_on_time' using binary encoding
data['completed_on_time'] = data['completed_on_time'].astype(int)

# Columns to normalize
columns_to_normalize = [
    "estimated_duration_x", "actual_duration", "quality_score",
    "success_score", "estimated_duration_y", "years_experience"
]

# Initialize and apply MinMaxScaler
scaler = MinMaxScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

ordinal_columns = ["skill_level", "complexity", "required_skill_level", "priority"]
data[ordinal_columns] = data[ordinal_columns].astype(int)  # Ensure they are integers


# Feature 1: Skill Match Score
data['skill_match_score'] = data.apply(
    lambda row: min(row['skill_level'] / row['required_skill_level'], 1), axis=1
)

# Feature 2: Experience-Adjusted Reliability
data['experience_adjusted_reliability'] = data.apply(
    lambda row: row['reliability_score'] * np.log1p(row['years_experience']) * row['avg_task_completion_rate'],
    axis=1
)
# Min-max normalization
min_reliability = data['experience_adjusted_reliability'].min()
max_reliability = data['experience_adjusted_reliability'].max()

data['experience_adjusted_reliability'] = (
    (data['experience_adjusted_reliability'] - min_reliability) / 
    (max_reliability - min_reliability)
)


# Drop the 'assignment_id' column
data = data.drop(columns=['assignment_id'])

# Replace spaces with underscores in all column names
data.columns = data.columns.str.replace(' ', '_')

# List of columns to convert
boolean_columns = data.select_dtypes(include=['bool']).columns.tolist()

# Convert TRUE/FALSE to 1/0
data[boolean_columns] = data[boolean_columns].astype(int)

# Display the updated dataset
data.to_csv("updated_construction_dataset.csv", index=False)


