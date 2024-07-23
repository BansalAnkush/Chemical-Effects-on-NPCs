import os
import numpy as np
import cv2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn import metrics
from scipy.stats import loguniform, uniform
from tqdm import tqdm
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

print("Importing libraries and setting up environment...")

BASE_DATA_FOLDER = "Benchmarking_Dataset"
TRAIN_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER)

def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    data = []
    labels = []
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
    return np.array(data), np.array(labels)

data, labels = load_and_preprocess_data()
print(f"Loaded {len(data)} images with {len(set(labels))} unique labels.")

# Optimized parameter space
param_dist = {
    'learning_rate': loguniform(1e-3, 1e-1),
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 5, 10],
    'max_iter': [50, 100, 150],
    'l2_regularization': loguniform(1e-6, 1e-3)
}

def run_model(iteration):
    print(f"Starting iteration {iteration + 1}")
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=0.1, random_state=42 + iteration)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.22, random_state=42 + iteration)

    print(f"Iteration {iteration + 1}: Performing halving random search...")
    gb = HistGradientBoostingClassifier(random_state=42 + iteration, early_stopping=True, validation_fraction=0.2, n_iter_no_change=10, tol=1e-4)
    
    halving_random_search = HalvingRandomSearchCV(
        estimator=gb,
        param_distributions=param_dist,
        n_candidates=5,
        cv=5,
        factor=3,
        random_state=42 + iteration,
        n_jobs=1
    )
    
    halving_random_search.fit(X_train, y_train)

    best_gb = halving_random_search.best_estimator_

    print(f"Iteration {iteration + 1}: Evaluating on validation set...")
    y_val_pred = best_gb.predict(X_val)
    val_accuracy = metrics.accuracy_score(y_val, y_val_pred)
    val_precision = metrics.precision_score(y_val, y_val_pred, average='macro')
    val_recall = metrics.recall_score(y_val, y_val_pred, average='macro')
    val_f1 = metrics.f1_score(y_val, y_val_pred, average='macro')

    print(f"Iteration {iteration + 1}: Evaluating on test set...")
    y_test_pred = best_gb.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    test_precision = metrics.precision_score(y_test, y_test_pred, average='macro')
    test_recall = metrics.recall_score(y_test, y_test_pred, average='macro')
    test_f1 = metrics.f1_score(y_test, y_test_pred, average='macro')

    print(f"Iteration {iteration + 1} completed.")
    return (val_accuracy, val_precision, val_recall, val_f1,
            test_accuracy, test_precision, test_recall, test_f1)

if __name__ == '__main__':
    n_iterations = 10
    print(f"Starting {n_iterations} parallel iterations...")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_model, range(n_iterations)), total=n_iterations, desc="Running iterations"))

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

    def mean_std(metric_list):
        mean = np.mean(metric_list)
        std = np.std(metric_list)
        if std == 0:
            std = np.finfo(float).eps
        return mean, std

    val_accuracy_mean, val_accuracy_std = mean_std(val_accuracies)
    val_precision_mean, val_precision_std = mean_std(val_precisions)
    val_recall_mean, val_recall_std = mean_std(val_recalls)
    val_f1_mean, val_f1_std = mean_std(val_f1_scores)

    test_accuracy_mean, test_accuracy_std = mean_std(test_accuracies)
    test_precision_mean, test_precision_std = mean_std(test_precisions)
    test_recall_mean, test_recall_std = mean_std(test_recalls)
    test_f1_mean, test_f1_std = mean_std(test_f1_scores)

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
        
        plt.savefig('hist_gradient_boosting_performance_metrics.png')
        print("Visualization saved as 'hist_gradient_boosting_performance_metrics.png'")
        plt.close()

    val_metrics = [(val_accuracy_mean, val_accuracy_std), (val_precision_mean, val_precision_std),
                   (val_recall_mean, val_recall_std), (val_f1_mean, val_f1_std)]
    test_metrics = [(test_accuracy_mean, test_accuracy_std), (test_precision_mean, test_precision_std),
                    (test_recall_mean, test_recall_std), (test_f1_mean, test_f1_std)]

    plot_metrics(val_metrics, test_metrics, 'HistGradient Boosting Model Performance Metrics')

    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Validation Mean': [val_accuracy_mean, val_precision_mean, val_recall_mean, val_f1_mean],
        'Validation SD': [val_accuracy_std, val_precision_std, val_recall_std, val_f1_std],
        'Test Mean': [test_accuracy_mean, test_precision_mean, test_recall_mean, test_f1_mean],
        'Test SD': [test_accuracy_std, test_precision_std, test_recall_std, test_f1_std]
    })

    metrics_df.to_csv('hist_gradient_boosting_performance_metrics.csv', index=False)
    print("Performance metrics saved as 'hist_gradient_boosting_performance_metrics.csv'")

    print("Process finished.")
