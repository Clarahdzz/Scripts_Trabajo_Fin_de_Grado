import pandas as pd

df = pd.read_csv("eeg_motor_imagery_entrenamiento_v2.csv")
print(df.head())
print(df["label"].value_counts())
print(df["trial"].value_counts())
print(df.columns)