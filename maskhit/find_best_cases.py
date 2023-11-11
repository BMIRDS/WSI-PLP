import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Read file into dataframe
df = pd.read_csv('predictions/ibd_project/2023_5_30-val-2-predictions.csv', header=0, index_col=0)
print(df)

# Apply sigmoid to the second prediction column to get probabilities for class 1
df["prob_class_1"] = sigmoid(df.iloc[:, 2].values)
df["prob_class_0"] = 1 - df["prob_class_1"]

# Extract columns
targets = df.iloc[:, 3]

# Define thresholds (modify as needed)
HIGH_CONFIDENCE_THRESHOLD = 0.75
HIGH_CONFIDENCE_THRESHOLD_0 = 0.6
MIDDLE_CONFIDENCE_UPPER = 0.55
MIDDLE_CONFIDENCE_LOWER = 0.45

# Identify cases based on conditions and return as dictionaries
def extract_cases(condition, prob_column):
    indices = df.index[condition].tolist()
    probabilities = df[prob_column][condition].tolist()
    return [{"index": idx, "probability": prob} for idx, prob in zip(indices, probabilities)]

# Identify cases
high_conf_tp_class_0 = extract_cases((df["prob_class_0"] > HIGH_CONFIDENCE_THRESHOLD_0) & (targets == 0), "prob_class_0")
high_conf_tp_class_1 = extract_cases((df["prob_class_1"] > HIGH_CONFIDENCE_THRESHOLD) & (targets == 1), "prob_class_1")
high_conf_fp_class_0 = extract_cases((df["prob_class_0"] > HIGH_CONFIDENCE_THRESHOLD_0) & (targets == 1), "prob_class_0")
high_conf_fp_class_1 = extract_cases((df["prob_class_1"] > HIGH_CONFIDENCE_THRESHOLD) & (targets == 0), "prob_class_1")
uncertain_cases = extract_cases((df["prob_class_0"] > MIDDLE_CONFIDENCE_LOWER) & 
                                (df["prob_class_0"] < MIDDLE_CONFIDENCE_UPPER) &
                                (df["prob_class_1"] > MIDDLE_CONFIDENCE_LOWER) & 
                                (df["prob_class_1"] < MIDDLE_CONFIDENCE_UPPER), 
                                "prob_class_0")  # or "prob_class_1", depending on which one you'd like to use

# Print results
print("High confidence true positives for class 0:", high_conf_tp_class_0)
print("High confidence true positives for class 1:", high_conf_tp_class_1)
print("High confidence false positives for class 0:", high_conf_fp_class_0)
print("High confidence false positives for class 1:", high_conf_fp_class_1)
print("Cases that were around 50-50 for both classes:", uncertain_cases)
