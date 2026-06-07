import pandas as pd
import numpy as np
import pickle
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

print("Loading keypoints.csv ...")
df = pd.read_csv("keypoints.csv")

print(f"Total samples  : {len(df)}")
print(f"Total classes  : {df['label'].nunique()}")

X = df.drop(columns=["label"]).values
y_raw = df["label"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True
    ))
])

print("\nTraining MLP Neural Network ")
pipeline.fit(X_train, y_train)

acc = pipeline.score(X_test, y_test)
print(f"\nTest Accuracy : {acc * 100:.2f}%")
print("\nPer class report:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs("model", exist_ok=True)
with open("model/yoga_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("\nModel saved!")
print("Training complete!")
