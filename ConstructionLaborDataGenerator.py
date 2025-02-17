import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_worker_data(num_workers=10):
    """Generate synthetic worker data"""
    skills = ['General Labor', 'Carpentry', 'Electrical', 'Plumbing', 'HVAC']
    
    workers = []
    for i in range(num_workers):
        worker = {
            'worker_id': f'W{i+1:03d}',
            'years_experience': np.random.randint(1, 20),
            'primary_skill': np.random.choice(skills),
            'skill_level': np.random.randint(1, 6),  # 1-5 rating
            'reliability_score': round(np.random.uniform(0.6, 1.0), 2),
            'avg_task_completion_rate': round(np.random.uniform(0.8, 1.2), 2)  # 1.0 is on time
        }
        workers.append(worker)
    
    return pd.DataFrame(workers)

def generate_task_data(num_tasks=10):
    """Generate synthetic construction task data"""
    task_types = ['Foundation Work', 'Framing', 'Electrical Installation', 
                  'Plumbing Installation', 'HVAC Installation', 'Finishing']
    
    tasks = []
    for i in range(num_tasks):
        task_type = np.random.choice(task_types)
        # Base duration depends on task type
        base_duration = {
            'Foundation Work': np.random.randint(15, 25),
            'Framing': np.random.randint(10, 20),
            'Electrical Installation': np.random.randint(5, 15),
            'Plumbing Installation': np.random.randint(5, 15),
            'HVAC Installation': np.random.randint(8, 16),
            'Finishing': np.random.randint(3, 10)
        }[task_type]
        
        task = {
            'task_id': f'T{i+1:03d}',
            'task_type': task_type,
            'required_skill_level': np.random.randint(1, 6),
            'estimated_duration': base_duration,
            'priority': np.random.randint(1, 4),  # 1 is highest priority
            'complexity': np.random.randint(1, 6)  # 1-5 scale
        }
        tasks.append(task)
    
    return pd.DataFrame(tasks)

def generate_historical_assignments(workers_df, tasks_df, num_assignments=50):
    """Generate historical task assignments and their outcomes"""
    assignments = []
    
    for i in range(num_assignments):
        # Randomly select worker and task
        worker = workers_df.iloc[np.random.randint(0, len(workers_df))]
        task = tasks_df.iloc[np.random.randint(0, len(tasks_df))]
        
        # Calculate actual duration based on various factors
        skill_match_factor = 1.0
        if task['task_type'].startswith(worker['primary_skill']):
            skill_match_factor = 0.9  # 10% faster if skills match
            
        experience_factor = max(0.8, min(1.2, 1.1 - (worker['years_experience'] * 0.01)))
        
        base_duration = task['estimated_duration']
        actual_duration = base_duration * skill_match_factor * experience_factor * worker['avg_task_completion_rate']
        
        # Add some random variation
        actual_duration *= np.random.uniform(0.9, 1.1)
        
        # Calculate success score
        time_efficiency = base_duration / actual_duration
        quality_score = np.random.uniform(
            max(0.7, worker['skill_level'] / 5), 
            min(1.0, (worker['skill_level'] + 1) / 5)
        )
        
        success_score = (time_efficiency * 0.6 + quality_score * 0.4) * 100
        
        assignment = {
            'assignment_id': f'A{i+1:03d}',
            'worker_id': worker['worker_id'],
            'task_id': task['task_id'],
            'estimated_duration': base_duration,
            'actual_duration': round(actual_duration, 1),
            'quality_score': round(quality_score * 100, 1),
            'success_score': round(success_score, 1),
            'completed_on_time': actual_duration <= base_duration * 1.1
        }
        assignments.append(assignment)
    
    return pd.DataFrame(assignments)

def generate_full_dataset(num_workers=10, num_tasks=10, num_assignments=500):
    """Generate complete dataset with workers, tasks, and historical assignments"""
    workers_df = generate_worker_data(num_workers)
    tasks_df = generate_task_data(num_tasks)
    assignments_df = generate_historical_assignments(workers_df, tasks_df, num_assignments)
    
    # Create merged dataset for training
    full_data = assignments_df.merge(workers_df, on='worker_id', how='left')
    full_data = full_data.merge(tasks_df, on='task_id', how='left')
    
    return workers_df, tasks_df, assignments_df, full_data

# Generate and save the data
workers, tasks, assignments, full_dataset = generate_full_dataset()

# Save to CSV files
workers.to_csv('construction_workers.csv', index=False)
tasks.to_csv('construction_tasks.csv', index=False)
assignments.to_csv('historical_assignments.csv', index=False)
full_dataset.to_csv('full_construction_dataset.csv', index=False)