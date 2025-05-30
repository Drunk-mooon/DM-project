
import pandas as pd
from sklearn.metrics import f1_score

# Load the CSV file
file_path = r"results.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)
encodings = ['utf-8', 'latin1', 'ISO-8859-1']
for enc in encodings:
    try:
        data1 = pd.read_csv(r"dataset\RU-Conflict\zero-shot\UKR\test.csv",nrows=150, encoding=enc, engine='python')
    except UnicodeDecodeError:
        continue
# Convert columns to strings
data['0'] = data['0'].astype(str)
data1['stance_label'] = data1['stance_label'].astype(str)

# Ensure only valid classes are considered
valid_labels = {"Oppose", "Support", "Neutral"}  # Three categories
data = data[data['0'].isin(valid_labels) & data1['stance_label'].isin(valid_labels)]

# Extract predictions and true labels
predicted_labels = data['0']
true_labels = data1['stance_label']

# Calculate Micro-F1 score for three categories
micro_f1 = f1_score(true_labels, predicted_labels, average='macro')

print("Micro-F1 Score for Three Categories:", micro_f1)
