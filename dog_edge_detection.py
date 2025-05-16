import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = 'Lena.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur with two different kernel sizes
blur1 = cv2.GaussianBlur(image, (3, 3), 0)
blur2 = cv2.GaussianBlur(image, (9, 9), 0)

# Compute Difference of Gaussian (DoG)
dog = cv2.absdiff(blur1, blur2)

# Normalize for better visualization
dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

# Plot the results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # Changed to 2x2 grid
axes = axes.ravel()  # Flatten axes array for easier indexing

titles = ["Original Image", "Blurred (3x3)", "Blurred (9x9)", "Difference of Gaussian (DoG)"]
images = [image, blur1, blur2, dog]

for i in range(4):  # Changed to range(4) to display all images
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()  # Uncommented this line for better spacing
plt.show()
