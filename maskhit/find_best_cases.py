import pandas as pd
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Read file into dataframe
df_full = pd.read_csv('predictions/ibd_project/2023_5_30_new-test-0-predictions.csv', header=0, index_col=0)

prediction_columns = df_full.columns[1:5]
class_probabilities = softmax(df_full[prediction_columns].values)

for i, column in enumerate(prediction_columns):
    df_full[f"prob_{column}"] = class_probabilities[:, i]

targets = df_full.iloc[:, 5]
targets = targets.astype(int)

CONFIDENCE_THRESHOLD = 0.9

df_full['predicted_class'] = np.argmax(class_probabilities, axis=1)
df_full['predicted_probability'] = np.max(class_probabilities, axis=1)

print(df_full)

# Identify high confidence true positives for each class
high_conf_tp = {}
for class_index in range(len(prediction_columns)):
    condition = (df_full['predicted_class'] == class_index) & (targets == class_index) & (df_full['predicted_probability'] > CONFIDENCE_THRESHOLD)
    filtered_df = df_full[condition]
    first_column_values = filtered_df.iloc[:, 0]
    indices = first_column_values.tolist()
    probabilities = df_full.loc[condition, 'predicted_probability'].tolist()

    # Pair each index with its corresponding probability
    high_conf_tp[class_index] = list(zip(indices, probabilities))

# Print results
for class_index, cases in high_conf_tp.items():
    if len(cases) > 0:
        highest_probability_case = max(cases, key=lambda x: x[1])
        print(f"For {class_index}: {highest_probability_case}")
    else:
        print("No confident cases found")
