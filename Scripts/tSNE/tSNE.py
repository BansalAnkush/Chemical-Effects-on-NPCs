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
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
import seaborn as sns
import random

# Unzip the data folder
!unzip TSNE_Analysis_edited_1.zip

# Set the base data folder path
BASE_DATA_FOLDER = "TSNE_Analysis_edited_1"
TRAIN_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER)

def visualize_scatter_with_images(X_2d_data, images, figsize=(45,45), image_zoom=1):
    """
    Visualize scatter plot with images.
    """
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.xlabel('tsne-2d-one')
    plt.ylabel('tsne-2d-two')
    plt.show()

def visualize_scatter(data_2d, label_ids, figsize=(40,40)):
    """
    Visualize scatter plot.
    """
    plt.figure(figsize=figsize)
    plt.grid()
    plt.xlabel('tSNE-1', fontsize=40)
    plt.ylabel('tSNE-2', fontsize=40)
    nb_classes = len(np.unique(label_ids))
    colors = ['#0000FF','#000000','#458B74','#CD3333','#838B8B','#458B74','#CD3333','#B23AEE','#B23AEE','#B23AEE','#B23AEE']
    markers = ['o' , '^' , 's' , 'D' , 'x' , 'D' , 'x' , 'x' , 'x' , 'o' , '^' , 's']
    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker=markers[label_id],
                    c=colors[label_id],
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')

# Load and preprocess the images and labels
images = []
labels = []
for class_folder_name in os.listdir(TRAIN_DATA_FOLDER):
    class_folder_path = os.path.join(TRAIN_DATA_FOLDER, class_folder_name)
    for image_path in glob(os.path.join(class_folder_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (45, 45))
        image = image.flatten()
        images.append(image)
        labels.append(class_folder_name)

images = np.array(images)
labels = np.array(labels)

# Create label to ID and ID to label dictionaries
label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])

# Scale the images
images_scaled = StandardScaler().fit_transform(images)

# Apply PCA
pca = PCA(n_components=2025)
pca_result = pca.fit_transform(images_scaled)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

# Visualize PCA results
plt.scatter(pca_result[:, 0], pca_result[:, 2024], c=label_ids)
plt.show()

# Apply t-SNE
tsne = TSNE(n_components=2, early_exaggeration=12.0, learning_rate=500.0,
            n_iter_without_progress=300, min_grad_norm=1e-7, metric="euclidean",
            init="random", verbose=0, random_state=42, method='barnes_hut', angle=0.5)
tsne_results = tsne.fit_transform(pca_result)

# Visualization settings
plt.figure(figsize=(40, 40))
given_colors = ['#0000FF', '#FF0000', '#0000FF', '#FF0000', '#0000FF', '#FF0000',
                '#0000FF', '#FF0000', '#0000FF', '#FF0000', '#0000FF', '#FF0000',
                '#0000FF', '#FF0000', '#0000FF', '#FF0000', '#0000FF', '#FF0000',
                '#0000FF', '#FF0000', '#0000FF', '#FF0000', '#0000FF', '#FF0000',
                '#0000FF', '#FF0000', '#0000FF', '#FF0000', '#0000FF', '#FF0000',
                '#0000FF', '#FF0000', '#0000FF', '#FF0000', '#0000FF', '#FF0000']

given_markers = ['D', 'D', 'D', 'D', 'D', 'D', '*', '*', 's', 's', 's', 's', 's', 's',
                 '^', '^', '^', '^', '^', '^', 'D', 'D', '*', '*', 's', 's', '^', '^',
                 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']

sizes = [400, 400, 100, 100, 200, 200, 400, 400, 400, 400, 100, 100, 200, 200,
         400, 400, 100, 100, 200, 200, 400, 400, 400, 400, 400, 400, 400, 400,
         400, 400, 400, 400, 100, 100, 200, 200]

# Indices at which no fill should be applied
no_fill_indices = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}

# Plot each class with its corresponding color, marker, and size
for i, (color, marker, size) in enumerate(zip(given_colors, given_markers, sizes)):
    if i >= len(np.unique(label_ids)):
        break
    if i in no_fill_indices:
        # Markers with only borders and no fill at specified indices
        plt.scatter(tsne_results[label_ids == i, 0], tsne_results[label_ids == i, 1],
                    edgecolor=color, facecolors='none', label=id_to_label_dict[i],
                    alpha=0.8, marker=marker, s=size)
    else:
        # Markers with fill color
        plt.scatter(tsne_results[label_ids == i, 0], tsne_results[label_ids == i, 1],
                    c=color, label=id_to_label_dict[i], alpha=0.8, marker=marker, s=size)

# Add legend
plt.legend(loc='best', fontsize=20)