import numpy as np
import json
import os
import argparse
import csv
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from skimage.metrics import structural_similarity as ssim

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory for results")
args = parser.parse_args()

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# Load dataset
digits = load_digits()
X, y = digits.images, digits.target
X = X.reshape(X.shape[0], -1) / 16.0  # Normalize

# Train models
models = {
    "RF": RandomForestClassifier(),
    "XGB": XGBClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)

# Define Evolutionary Algorithm with DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Prepare log directory
log_dir = os.path.join(out_dir, "logs")
os.makedirs(log_dir, exist_ok=True)
csv_file_path = os.path.join(log_dir, "experiment_results.csv")

# Initialize CSV file
with open(csv_file_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Class", "Generation", "Confidence", "SSIM"])

final_images = {}  # Store final images for canvas

def get_median_image(target_class):
    """Compute the median image for the given class from the dataset."""
    class_images = X[y == target_class]
    return np.median(class_images, axis=0).reshape(8, 8)

def evaluate(individual, model, target_class, median_image):
    """Evaluate confidence and SSIM against the median class image."""
    img_flat = np.array(individual).flatten().reshape(1, -1)
    confidence = model.predict_proba(img_flat)[0][target_class]
    
    # Compute SSIM against the median image of the class
    ssim_score = ssim(np.array(individual).reshape(8, 8), median_image, data_range=1.0)

    return confidence, ssim_score

def mutate(individual, mu=0, sigma=0.1):
    noise = np.random.normal(mu, sigma, len(individual))
    individual[:] = np.clip(individual + noise, 0, 1)
    return individual,

def evolve_images(model, model_name, target_class, generations=500, population_size=10):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=64)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, model=model, target_class=target_class, median_image=get_median_image(target_class))
    toolbox.register("mate", tools.cxBlend, alpha=0.2)
    toolbox.register("mutate", mutate, mu=0, sigma=0.1)
    
    # Use tournament selection
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=population_size)
    best_confidence = 0.0
    best_image = None
    best_ssim = 0.0

    for gen in range(1, generations + 1):  # Start from generation 1
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit[0],)

        best_individual = tools.selBest(population, 1)[0]
        best_gen_conf = best_individual.fitness.values[0]
        best_gen_ssim = evaluate(best_individual, model, target_class, get_median_image(target_class))[1]

        if best_gen_conf > best_confidence:
            best_confidence = best_gen_conf
            best_image = np.array(best_individual).reshape(8, 8).astype(np.float32)
            best_ssim = best_gen_ssim  # Ensuring SSIM corresponds to the best confidence image

        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = (fit[0],)
        
        population[:] = toolbox.select(offspring, k=len(population))

        # Log every generation to CSV
        with open(csv_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_name, target_class, gen, best_confidence, best_ssim])

    # Store final evolved image for canvas
    final_images[(model_name, target_class)] = (best_image, best_confidence, best_ssim)

    return {"best_confidence": best_confidence, "best_ssim": best_ssim}

# Run evolution for each class and model
results = {}
for model_name, model in models.items():
    results[model_name] = {}
    for class_label in range(10):
        results[model_name][class_label] = evolve_images(model, model_name, class_label)

# Reformat results to include "means" for AI-Scientist compatibility
formatted_results = {model: {"means": results[model]} for model in results}

# Save final best results in AI Scientist format
result_path = os.path.join(out_dir, "final_info.json")
with open(result_path, "w") as f:
    json.dump(formatted_results, f, indent=4)



# Filter valid images for the canvas
valid_images = {k: v for k, v in final_images.items() if v[0] is not None}

# Create canvas for final images
num_models = len(models)
fig, axes = plt.subplots(num_models, 10, figsize=(15, num_models * 2))

for (model_name, class_label), (image, conf, ssim_score) in valid_images.items():
    row = list(models.keys()).index(model_name)
    col = class_label

    axes[row, col].imshow(image, cmap="gray")
    axes[row, col].axis("off")
    axes[row, col].set_title(
        f"{model_name} / {class_label}\nConf: {conf:.4f} / SSIM: {ssim_score:.4f}",
        fontsize=7, pad=3
    )

# Adjust layout
plt.tight_layout()
canvas_path = os.path.join(log_dir, "final_results.png")
plt.savefig(canvas_path, bbox_inches="tight", pad_inches=0.2)
plt.close()

print(f"Experiment completed. Results saved to {result_path} and {csv_file_path}")
