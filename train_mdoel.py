import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/processed_dataset.csv")

df.columns = df.columns.str.lower()

X = df.drop("disease", axis=1)
y = df["disease"]
model = RandomForestClassifier(n_estimators=200)
model.fit(X, y)

joblib.dump(model, "models/disease_model.pkl")

print("Disease model trained!")