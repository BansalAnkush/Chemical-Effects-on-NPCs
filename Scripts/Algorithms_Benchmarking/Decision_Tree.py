# -*- coding: utf-8 -*-
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from tqdm import tqdm
import multiprocessing
import pandas as pd

print("Importing libraries and setting up environment...")

BASE_DATA_FOLDER = "Benchmarking_Dataset"
TRAIN_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER)

print("Loading and preprocessing data...")

# Initialize data and label arrays
data = []
labels = []

# Loop through each subfolder in Benchmarking_Dataset
for subfolder in os.listdir(TRAIN_DATA_FOLDER):
    subfolder_path = os.path.join(TRAIN_DATA_FOLDER, subfolder)
    if os.path.isdir(subfolder_path):
        for image_name in os.listdir(subfolder_path):
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(subfolder_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (100, 100)).flatten()
            data.append(img)
            labels.append(subfolder)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print(f"Loaded {len(data)} images with {len(set(labels))} unique labels.")

# Function to run a single iteration of model training and evaluation
def run_model(iteration):
    print(f"Starting iteration {iteration + 1}")
    
    # First split: 90% train+val, 10% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=0.1, random_state=42 + iteration)

    # Second split: 70% train, 20% val (0.22 of 0.9 is roughly 0.2 of the total)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.22, random_state=42 + iteration)

    print(f"Iteration {iteration + 1}: Performing grid search...")
    clf = DecisionTreeClassifier()
    param_grid = {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_

    print(f"Iteration {iteration + 1}: Evaluating on validation set...")
    # Evaluate on validation set
    y_val_pred = best_clf.predict(X_val)
    val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
    val_precision = metrics.precision_score(y_val, y_val_pred, average='macro')
    val_recall = metrics.recall_score(y_val, y_val_pred, average='macro')
    val_f1 = metrics.f1_score(y_val, y_val_pred, average='macro')

    print(f"Iteration {iteration + 1}: Evaluating on test set...")
    # Evaluate on test set
    y_test_pred = best_clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_precision = metrics.precision_score(y_test, y_test_pred, average='macro')
    test_recall = metrics.recall_score(y_test, y_test_pred, average='macro')
    test_f1 = metrics.f1_score(y_test, y_test_pred, average='macro')

    print(f"Iteration {iteration + 1} completed.")
    return (val_accuracy, val_precision, val_recall, val_f1,
            test_accuracy, test_precision, test_recall, test_f1)

if __name__ == '__main__':
    # Number of iterations
    n_iterations = 10

    print(f"Starting {n_iterations} parallel iterations...")

    # Use multiprocessing to run iterations in parallel
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(run_model, range(n_iterations)), total=n_iterations, desc="Running iterations"))

    # Unpack results
    val_accuracies, val_precisions, val_recalls, val_f1_scores = [], [], [], []
    test_accuracies, test_precisions, test_recalls, test_f1_scores = [], [], [], []

    for result in results:
        val_accuracies.append(result[0])
        val_precisions.append(result[1])
        val_recalls.append(result[2])
        val_f1_scores.append(result[3])
        test_accuracies.append(result[4])
        test_precisions.append(result[5])
        test_recalls.append(result[6])
        test_f1_scores.append(result[7])

    print("All iterations completed. Calculating statistics...")

    # Calculate mean and standard deviation
    def mean_std(metric_list):
        mean = np.mean(metric_list)
        std = np.std(metric_list)
        if std == 0:
            std = np.finfo(float).eps  # Use machine epsilon if std is zero
        return mean, std

    val_accuracy_mean, val_accuracy_std = mean_std(val_accuracies)
    val_precision_mean, val_precision_std = mean_std(val_precisions)
    val_recall_mean, val_recall_std = mean_std(val_recalls)
    val_f1_mean, val_f1_std = mean_std(val_f1_scores)

    test_accuracy_mean, test_accuracy_std = mean_std(test_accuracies)
    test_precision_mean, test_precision_std = mean_std(test_precisions)
    test_recall_mean, test_recall_std = mean_std(test_recalls)
    test_f1_mean, test_f1_std = mean_std(test_f1_scores)

    # Print results
    print("\nFinal Results:")
    print("Validation Set Results:")
    print(f"Accuracy: {val_accuracy_mean:.4f} (±{val_accuracy_std:.4f})")
    print(f"Precision: {val_precision_mean:.4f} (±{val_precision_std:.4f})")
    print(f"Recall: {val_recall_mean:.4f} (±{val_recall_std:.4f})")
    print(f"F1 Score: {val_f1_mean:.4f} (±{val_f1_std:.4f})")

    print("\nTest Set Results:")
    print(f"Accuracy: {test_accuracy_mean:.4f} (±{test_accuracy_std:.4f})")
    print(f"Precision: {test_precision_mean:.4f} (±{test_precision_std:.4f})")
    print(f"Recall: {test_recall_mean:.4f} (±{test_recall_std:.4f})")
    print(f"F1 Score: {test_f1_mean:.4f} (±{test_f1_std:.4f})")

    print("\nGenerating visualization...")

    # Visualize results
    def plot_metrics(val_metrics, test_metrics, title):
        plt.figure(figsize=(12, 6))
        x = np.arange(4)
        width = 0.35

        plt.bar(x - width/2, [m[0] for m in val_metrics], width, label='Validation', yerr=[m[1] for m in val_metrics], capsize=5)
        plt.bar(x + width/2, [m[0] for m in test_metrics], width, label='Test', yerr=[m[1] for m in test_metrics], capsize=5)

        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        plt.legend()
        plt.ylim(0, 1)
        
        # Save the plot as an image
        plt.savefig('model_performance_metrics.png')
        print("Visualization saved as 'model_performance_metrics.png'")
        plt.close()

    val_metrics = [(val_accuracy_mean, val_accuracy_std), (val_precision_mean, val_precision_std),
                   (val_recall_mean, val_recall_std), (val_f1_mean, val_f1_std)]
    test_metrics = [(test_accuracy_mean, test_accuracy_std), (test_precision_mean, test_precision_std),
                    (test_recall_mean, test_recall_std), (test_f1_mean, test_f1_std)]

    plot_metrics(val_metrics, test_metrics, 'Model Performance Metrics')

    # Export performance metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Validation Mean': [val_accuracy_mean, val_precision_mean, val_recall_mean, val_f1_mean],
        'Validation SD': [val_accuracy_std, val_precision_std, val_recall_std, val_f1_std],
        'Test Mean': [test_accuracy_mean, test_precision_mean, test_recall_mean, test_f1_mean],
        'Test SD': [test_accuracy_std, test_precision_std, test_recall_std, test_f1_std]
    })

    metrics_df.to_csv('performance_metrics.csv', index=False)
    print("Performance metrics saved as 'performance_metrics.csv'")

    print("Process finished.")
