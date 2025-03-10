import argparse
import os
import json
import sys
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Parse command-line arguments properly
parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory for results")
args = parser.parse_args()

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# 📊 Load Penguins dataset
df = sns.load_dataset("penguins").dropna()
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = df["species"]

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔄 Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Define classifiers
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

# 🚀 Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"means": acc}  # ✅ AI Scientist expects "means" format

# 📝 Save results
result_path = os.path.join(out_dir, "final_info.json")
with open(result_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"Experiment completed. Results saved to {result_path}")
