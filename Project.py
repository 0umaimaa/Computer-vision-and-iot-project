import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_colors(image, k=5):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return colors

def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_colors(colors):
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        plt.bar(i, 1, color=color/255, edgecolor='white')
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    image_path = r'C:\Users\HP\Documents\pic1.jpg'  # Use a raw string
    image = load_image(image_path)
    display_image(image)
    dominant_colors = extract_colors(image)
    display_colors(dominant_colors)
