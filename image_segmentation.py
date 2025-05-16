import numpy as np
import cv2
import matplotlib.pyplot as plt

def k_means_image_segmentation(image, k, max_iter=100):

    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.2)

    _, label, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_image = centers[label.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, label, centers

image = cv2.imread('messi.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

k = 3

segmented_image, labels, centers = k_means_image_segmentation(image, k)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title(f"Segmented Image with {k} Clusters")
plt.imshow(segmented_image)

plt.show()
