import cv2
import numpy as np
import matplotlib.pyplot as plt

def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))

def reflect_image(image, axis='x'):
    if axis == 'x':
        return cv2.flip(image, 0)  # Flip vertically
    elif axis == 'y':
        return cv2.flip(image, 1)  # Flip horizontally
    else:
        return cv2.flip(image, -1)  # Flip both axes

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def scale_image(image, fx, fy):
    return cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

def crop_image(image, x, y, w, h):
    return image[y:y+h, x:x+w]

def shear_image(image, shear_x=0, shear_y=0):
    rows, cols = image.shape[:2]
    M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

# Load an image
image = cv2.imread('messi.webp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply transformations
translated = translate_image(image, 50, 30)
reflected = reflect_image(image, 'y')
rotated = rotate_image(image, 45)
scaled = scale_image(image, 1.5, 1.5)
cropped = crop_image(image, 50, 50, 200, 200)
sheared_x = shear_image(image, 0, 0.1)
sheared_y = shear_image(image, 0.2, 0)

# Display all images using matplotlib
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.ravel()

titles = ['Original Image', 'Translated', 'Reflected', 'Rotated', 'Scaled', 'Cropped', 'Sheared X-axis', 'Sheared Y-axis']
images = [image, translated, reflected, rotated, scaled, cropped, sheared_x, sheared_y]

for i in range(len(images)):
    if len(images[i].shape) == 3:
        axes[i].imshow(images[i], cmap='gray')
    else:
        axes[i].imshow(images[i])
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()