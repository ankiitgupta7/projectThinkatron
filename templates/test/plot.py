import json
import matplotlib.pyplot as plt

# 📂 Load results
with open("results/final_info.json", "r") as f:
    results = json.load(f)

# 📊 Plot results
plt.bar(results.keys(), results.values(), color=["blue", "green", "red"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Penguins Classification Model Performance")
plt.ylim(0, 1)
plt.show()
