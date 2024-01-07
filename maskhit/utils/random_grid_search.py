import pandas as pd
import yaml
import os
import subprocess
import random  # For random sampling
from pathlib import Path

# ------------------------------
# Random Grid Search Hyperparameter Tuning
# Example Usage: CUDA_VISIBLE_DEVICES=8 python grid.py
# ------------------------------

# Define lists of hyperparameter values to experiment with
learning_rates_preds = [5e-1, 5e-2, 5e-3, 5e-4, 5e-5, 5e-6]
weight_decay_preds = [1e-1, 1e-2, 1e-3, 1e-4]
dropouts = [0.05, 0.1, 0.2, 0.3, 0.4]
accumulation_steps_list = [1, 2]
batch_sizes = [8]

# Initialize an empty DataFrame to store results of all combinations
global_df = pd.DataFrame(columns=['lr_pred', 'wd_pred', 'dropout', 'accum_steps', 'batch_sizes', 'f1_zero', 'auc_zero', 'loss_zero', 'f1_two', 'auc_two', 'loss_two', 'f1_four', 'auc_four', 'loss_four'])

# Ensure a directory exists for temporary configuration files
if not os.path.exists("temp"):
    os.makedirs("temp")

# Load the base configuration file as a template
with open("configs/config_ibd_train.yml", "r") as file:
    config_template = yaml.safe_load(file)

# Number of random combinations to sample
NUM_TRIALS = 16  # Define the number of random trials

# Sample and iterate over random combinations of hyperparameters
for _ in range(NUM_TRIALS):
    lr_pred = random.choice(learning_rates_preds)
    wd_pred = random.choice(weight_decay_preds)
    dropout = random.choice(dropouts)
    accum_steps = random.choice(accumulation_steps_list)
    batch_size = random.choice(batch_sizes)

    # Modify the base config with the current combination of hyperparameters
    temp_config = config_template.copy()
    temp_config['model']['lr_pred'] = lr_pred
    temp_config['model']['wd_pred'] = wd_pred
    temp_config['model']['dropout'] = dropout
    temp_config['model']['accumulation_steps'] = accum_steps
    temp_config['model']['batch_size'] = batch_size

    # Define a unique filename for the current config
    file_name = f"config_temp_{lr_pred}_{wd_pred}_{dropout}_{accum_steps}_{batch_size}.yml"
    temp_file_path = Path("temp") / file_name

    # Save the modified config to a temporary file
    with open(temp_file_path, "w") as file:
        yaml.dump(temp_config, file)

    # Execute the model training with the current configuration
    cmd = [
        'python', 'cross_validation.py',
        '--user-config-file', temp_file_path,
        '--default-config-file', 'configs/config_default.yaml',
        '--folds', '0, 2, 4'
    ]
    print(cmd)
    subprocess.run(cmd)

    # Load the results and compute the averages for fold 0
    df = pd.read_csv('logs/ibd_project/2023_5_30-test-0_data.csv')
    avg_f1 = df['f1'].mean()
    avg_auc = df['auc'].mean()
    avg_loss = df['loss'].mean()

    # Load the results and compute the averages for fold 2
    df_2 = pd.read_csv('logs/ibd_project/2023_5_30-test-2_data.csv')
    avg_f1_2 = df_2['f1'].mean()
    avg_auc_2 = df_2['auc'].mean()
    avg_loss_2 = df_2['loss'].mean()
    
    # Load the results and compute the averages for fold 4
    df_4 = pd.read_csv('logs/ibd_project/2023_5_30-test-4_data.csv')
    avg_f1_4 = df_4['f1'].mean()
    avg_auc_4 = df_4['auc'].mean()
    avg_loss_4 = df_4['loss'].mean()

    # Store the results along with the hyperparameters in the global dataframe
    row = {
        'lr_pred': lr_pred,
        'wd_pred': wd_pred,
        'dropout': dropout,
        'accum_steps': accum_steps,
        'batch_sizes': batch_size,
        'f1_zero': avg_f1,
        'auc_zero': avg_auc,
        'loss_zero': avg_loss,
        'f1_two': avg_f1_2,
        'auc_two': avg_auc_2,
        'loss_two': avg_loss_2,
        'f1_four': avg_f1_4,
        'auc_four': avg_auc_4,
        'loss_four': avg_loss_4
    }
    
    global_df = global_df.append(row, ignore_index=True)

    global_df.to_csv('random_grid_log.csv', index=False)

    # Clean up: Remove the temporary configuration file
    os.remove(temp_file_path)

# Display the results of all combinations
print(global_df)
