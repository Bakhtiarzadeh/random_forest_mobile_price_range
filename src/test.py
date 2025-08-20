import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = joblib.load('random_forest_model.joblib')

df = pd.read_csv('../data/test.csv')

target_column = 'price_range'

x_test = df.drop(columns=[target_column])
y_test = df[[target_column]]

X_test = x_test.to_numpy()
y_test = y_test.to_numpy()
y_test = y_test.ravel()

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)