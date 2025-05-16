import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('skyline.webp', cv2.IMREAD_GRAYSCALE)

prewitt_kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y)
prewitt_combined = np.hypot(prewitt_x, prewitt_y)
prewitt_combined_mask = np.sqrt(np.square(prewitt_x) + np.square(prewitt_y))

sobel_x = cv2.filter2D(image, -1, sobel_kernel_x)
sobel_y = cv2.filter2D(image, -1, sobel_kernel_y)
sobel_combined = np.hypot(sobel_x, sobel_y)
sobel_combined_mask = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

plt.figure(figsize=(12, 14))

plt.subplot(3, 4, 2)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(3, 4, 5)
plt.imshow(prewitt_x, cmap='gray')
plt.title('Prewitt X')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(prewitt_y, cmap='gray')
plt.title('Prewitt Y')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(prewitt_combined, cmap='gray')
plt.title('Prewitt Combined')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(prewitt_combined_mask, cmap='gray')
plt.title('Prewitt Combined Mask')
plt.axis('off')

plt.subplot(3, 4, 9)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Combined')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(sobel_combined_mask, cmap='gray')
plt.title('Sobel Combined Mask')
plt.axis('off')

plt.show()
