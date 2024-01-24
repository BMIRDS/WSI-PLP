import pandas as pd
import yaml
import os
import subprocess

# ------------------------------
# Hard-coded Grid Search Hyperparameter Tuning
# ------------------------------

# Define lists of hyperparameter values to experiment with
hard_coded = [
    [7e-5, 0.007, 0.15, 2, 8],
    [0.001, 0.05, 0.1, 2, 4],
    [1e-5, 0.0001, 0.25, 2, 8],
    [7e-5, 0.002, 0.05, 2, 8],
    [1e-5, 0.3, 0.25, 1, 4],
    [0.0005, 0.003, 0.1, 2, 8],
    [0.005, 0.07, 0.3, 1, 8],
    [3e-5, 0.07, 0.25, 2, 4],
    [0.003, 0.3, 0.25, 1, 8]
]


# Initialize an empty DataFrame to store results of all combinations
global_df = pd.DataFrame(columns=['lr_pred', 'wd_pred', 'dropout', 'accum_steps', 'batch_sizes', 'f1_one', 'auc_one', 'loss_one', 'f1_three', 'auc_three', 'loss_three'])

# Ensure a directory exists for temporary configuration files
if not os.path.exists("temp"):
    os.makedirs("temp")

# Load the base configuration file as a template
with open("configs/config_ibd_train.yml", "r") as file:
    config_template = yaml.safe_load(file)

# Iterate over each combination in the hard-coded list
for comb in hard_coded:
    lr_pred, wd_pred, dropout, accum_steps, batch_size = comb

    # Modify the base config with the current combination of hyperparameters
    temp_config = config_template.copy()
    temp_config['model']['lr_pred'] = lr_pred
    temp_config['model']['wd_pred'] = wd_pred
    temp_config['model']['dropout'] = dropout
    temp_config['model']['accumulation_steps'] = accum_steps
    temp_config['model']['batch_size'] = batch_size

    # Define a unique filename for the current config
    temp_file_path = os.path.join("temp", f"config_temp_{lr_pred}_{wd_pred}_{dropout}_{accum_steps}_{batch_size}.yml")

    # Save the modified config to a temporary file
    with open(temp_file_path, "w") as file:
        yaml.dump(temp_config, file)

    # Execute the model training with the current configuration
    cmd = [
        'python', 'cross_validation.py',
        '--user-config-file', temp_file_path,
        '--default-config-file', 'configs/config_default.yaml',
        '--folds', '1, 3'
    ]
    subprocess.run(cmd)

    # Load the results and compute the averages for fold 1
    df_1 = pd.read_csv('logs/ibd_project/2023_5_30-test-1_data.csv')
    avg_f1_1 = df_1['f1'].mean()
    avg_auc_1 = df_1['auc'].mean()
    avg_loss_1 = df_1['loss'].mean()

    # Load the results and compute the averages for fold 3
    df_3 = pd.read_csv('logs/ibd_project/2023_5_30-test-3_data.csv')
    avg_f1_3 = df_3['f1'].mean()
    avg_auc_3 = df_3['auc'].mean()
    avg_loss_3 = df_3['loss'].mean()

    # Store the results along with the hyperparameters in the global dataframe
    row = {
        'lr_pred': lr_pred,
        'wd_pred': wd_pred,
        'dropout': dropout,
        'accum_steps': accum_steps,
        'batch_sizes': batch_size,
        'f1_one': avg_f1_1,
        'auc_one': avg_auc_1,
        'loss_one': avg_loss_1,
        'f1_three': avg_f1_3,
        'auc_three': avg_auc_3,
        'loss_three': avg_loss_3
    }
    
    global_df = global_df.append(row, ignore_index=True)

    global_df.to_csv('hard_grid_log.csv', index=False)

    # Clean up: Remove the temporary configuration file
    os.remove(temp_file_path)

# Display the results of all combinations
print(global_df)
