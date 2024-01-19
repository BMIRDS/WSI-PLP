# Example Usage: python plot_results.py logs/ibd_project/2023_5_30-0_data.csv

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

# setting up argument parser
parser = argparse.ArgumentParser(description='Plot AUC, F1, and loss curves')
parser.add_argument('filename', type=str, help='Path to the CSV file containing the data.')
args = parser.parse_args()

# reading the log file
try:
    df = pd.read_csv(args.filename, sep='\t', header=None)
except Exception as e:
    print(f"Error reading file {args.filename}")
    sys.exit(1)  # Exits the program

# 6 columns are present in the database with sep condition
df.columns = ['auc', 'epoch', 'f1', 'loss', 'mode', 'other']

# Splitting the column values
for col in df.columns:
    # last column is useless
    if col != 'other':
        df[col] = df[col].apply(lambda x: x.split(': ')[1])

# adding values to the dataframe
df['auc'] = df['auc'].astype(float)
df['epoch'] = df['epoch'].astype(int)
df['f1'] = df['f1'].astype(float)
df['loss'] = df['loss'].astype(float)
df['mode'] = df['mode'].astype(str)

# Splitting the DataFrame into training and validation
train_df = df[df["mode"] == "train"]
val_df = df[df["mode"] == "val"]

# Plotting the data
plt.figure(figsize=(15, 5))  # Adjusted figure size for a horizontal layout

# Plotting AUC (first plot)
plt.subplot(1, 3, 1)
plt.plot(train_df["epoch"], train_df["auc"], label="Train AUC")
plt.plot(val_df["epoch"], val_df["auc"], label="Val AUC")
plt.title("Training & Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()

# Plotting F1 (second plot)
plt.subplot(1, 3, 2) 
plt.plot(train_df["epoch"], train_df["f1"], label="Train F1")
plt.plot(val_df["epoch"], val_df["f1"], label="Val F1")
plt.title("Training & Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()

# Plotting Loss (third plot)
plt.subplot(1, 3, 3)
plt.plot(train_df["epoch"], train_df["loss"], label="Train Loss")
plt.plot(val_df["epoch"], val_df["loss"], label="Val Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig('results.png')
