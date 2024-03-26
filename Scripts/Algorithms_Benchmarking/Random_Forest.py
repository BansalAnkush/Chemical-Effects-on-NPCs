# %matplotlib inline
import pandas.util.testing as tm
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as ss
import seaborn as sns
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

!unzip Benchmarking.zip
BASE_DATA_FOLDER = "TSNE_Analysis"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER)

# Initialize data and label arrays
data = []
labels = []

# Loop through each subfolder in TSNE_Analysis
for subfolder in os.listdir('TSNE_Analysis'):
    subfolder_path = os.path.join('TSNE_Analysis', subfolder)
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Initialize Random Forest Classifier
clf = RandomForestClassifier()

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform grid search
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model
best_clf = grid_search.best_estimator_

# Prediction using the best model
y_pred = best_clf.predict(X_test)
y_prob = best_clf.predict_proba(X_test)

# Compute the metrics
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, average='macro'))
print("Recall:", metrics.recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", metrics.f1_score(y_test, y_pred, average='macro'))