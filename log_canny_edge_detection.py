import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'houses.bmp'  # Replace with your image path
og_image = cv2.imread(image_path)
og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur for better edge detection
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Convert to absolute and uint8 for display
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_combined = cv2.convertScaleAbs(sobel_combined)

# Laplacian Edge Detection
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)

# Laplacian of Gaussian (LoG)
log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
log = cv2.convertScaleAbs(log)

# Canny Edge Detection
canny = cv2.Canny(image, 100, 200)

# Plot results
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
axes = axes.ravel()

titles = [
    "OG_Original", "Original", "Sobel X", "Sobel Y", "Sobel Combined",
    "Laplacian", "LoG", "Canny"
]
images = [og_image, image, sobel_x, sobel_y, sobel_combined, laplacian, log, canny]

for i in range(len(images)):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(titles[i])
    axes[i].axis('off')

# plt.tight_layout()
plt.show()
