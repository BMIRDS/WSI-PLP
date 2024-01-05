"""
This script processes a CSV file containing predictions produced by running train.py in test mode.
The scripts applies the softmax function to calculate class probabilities. It identifies high-confidence true positives
and high-confidence false positives for each class, based on a defined confidence threshold that can be set by the user.
The script uses argparse to allow for file input. Results for true positive and false positive are sorted by confidence and displayed 
showcasing the top cases for each class.
"""

import pandas as pd
import numpy as np
import argparse

# Global Variables
CONFIDENCE_THRESHOLD = 0.5

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def display_result(df, label):
    print(label)
    for class_index, cases in df.items():
        if len(cases) > 0:
            # Sort the cases by probability in descending order and select the top 3
            top_cases = sorted(cases, key=lambda x: x[1], reverse=True)[:10]
            print(f"For {class_index}: {top_cases}")
        else:
            print(f"For {class_index}: No confident cases found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a CSV file.')
    parser.add_argument('--file', default='predictions/ibd_project/2023_5_30_new-test-0-predictions.csv', 
                        help='Path to the CSV predictions file')
    args = parser.parse_args()

    # Read file into dataframe ()
    df_full = pd.read_csv(args.file, header=0, index_col=0)

    prediction_columns = df_full.columns[1:5]
    class_probabilities = softmax(df_full[prediction_columns].values)

    for i, column in enumerate(prediction_columns):
        df_full[f"prob_{column}"] = class_probabilities[:, i]

    targets = df_full.iloc[:, 5]
    targets = targets.astype(int)

    df_full['predicted_class'] = np.argmax(class_probabilities, axis=1)
    df_full['predicted_probability'] = np.max(class_probabilities, axis=1)

    # Identify high confidence true positives and false positives for each class
    high_conf_tp = {}
    high_conf_fp = {}

    for class_index in range(len(prediction_columns)):
        # for true positives
        condition_tp = (df_full['predicted_class'] == class_index) & (targets == class_index) & (df_full['predicted_probability'] > CONFIDENCE_THRESHOLD)
        # for false positives
        condition_fp = (df_full['predicted_class'] != class_index) & (targets == class_index) & (df_full['predicted_probability'] > CONFIDENCE_THRESHOLD)
        
        filtered_df_tp = df_full[condition_tp]
        filtered_df_fp = df_full[condition_fp]
        
        first_column_values_tp = filtered_df_tp.iloc[:, 0]
        first_column_values_fp = filtered_df_fp.iloc[:, 0]

        indices_tp = first_column_values_tp.tolist()
        indices_fp = first_column_values_fp.tolist()

        probabilities_tp = df_full.loc[condition_tp, 'predicted_probability'].tolist()
        probabilities_fp = df_full.loc[condition_fp, 'predicted_probability'].tolist()

        # Pair each index with its corresponding probability
        high_conf_tp[class_index] = list(zip(indices_tp, probabilities_tp))
        high_conf_fp[class_index] = list(zip(indices_fp, probabilities_fp))

    # Print results for confident true positives
    display_result(high_conf_tp, 'true positive')
    display_result(high_conf_fp, 'false positive')
