import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot experiment results")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory for results")
args = parser.parse_args()

out_dir = args.out_dir
log_dir = os.path.join(out_dir, "logs")
plot_dir = os.path.join(log_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Load experiment results CSV
csv_file_path = os.path.join(log_dir, "experiment_results.csv")
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"Experiment results CSV not found at {csv_file_path}")

df = pd.read_csv(csv_file_path)

# Plot confidence vs. SSIM for each model
models = df["Model"].unique()

# Create individual plots for each model
for model in models:
    plt.figure(figsize=(10, 6))
    
    model_df = df[df["Model"] == model]
    
    for class_label in range(10):
        class_df = model_df[model_df["Class"] == class_label]
        plt.plot(class_df["Confidence"], class_df["SSIM"], marker="o", linestyle="-", label=f"Class {class_label}")

    plt.xlabel("Model Confidence")
    plt.ylabel("SSIM")
    plt.title(f"Confidence vs. SSIM for {model}")
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(plot_dir, f"{model}_confidence_vs_ssim.png")
    plt.savefig(plot_path, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    print(f"Saved plot: {plot_path}")

# Create a combined plot for all models
plt.figure(figsize=(10, 6))

for model in models:
    model_df = df[df["Model"] == model]
    avg_confidence = model_df.groupby("Generation")["Confidence"].mean()
    avg_ssim = model_df.groupby("Generation")["SSIM"].mean()

    plt.plot(avg_confidence, avg_ssim, marker="o", linestyle="-", label=f"{model}")

plt.xlabel("Model Confidence")
plt.ylabel("SSIM")
plt.title("Overall Confidence vs. SSIM for All Models")
plt.legend()
plt.grid(True)

# Save combined plot
combined_plot_path = os.path.join(plot_dir, "all_models_confidence_vs_ssim.png")
plt.savefig(combined_plot_path, bbox_inches="tight", pad_inches=0.2)
plt.close()
print(f"Saved combined plot: {combined_plot_path}")
