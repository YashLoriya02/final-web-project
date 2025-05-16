import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./EXP1/messi.jpg')
image2 = cv2.imread('./EXP1/goldhill.bmp')

image2 = cv2.resize(image2, (image.shape[1], image.shape[0]))

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Shape: {image.shape}")
print(f"Size: {image.size}")
print(f"Datatype: {image.dtype}")

b, g, r = cv2.split(image)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

h, w, _ = image.shape
cropped_image = image[h//6:5*h//6, w//6:5*w//6]

bright_image = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * 50)

inverted_image = cv2.bitwise_not(image)
and_image = cv2.bitwise_and(image, image2)
or_image = cv2.bitwise_or(image, image2)
xor_image = cv2.bitwise_xor(image, image2)

horizontal_flip = cv2.flip(image, 1)
vertical_flip = cv2.flip(image, 0)

matrix = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
perspective_image = cv2.warpPerspective(image, matrix, (int(w*1.5), int(h*1.5)))

fig, axes = plt.subplots(4, 4, figsize=(18, 9))
axes = axes.ravel()

titles = ['Original Image', 'Red Channel', 'Green Channel', 'Blue Channel',
            'Grayscale Image', 'Cropped Image', 'Brightened Image', 'Inverted Image', 'Image 2',
            'Bitwise AND', 'Bitwise OR', 'Bitwise XOR', 'Horizontal Flip', 'Vertical Flip',
            'Perspective Transform']

images = [image, b, g, r, gray_image, cropped_image, bright_image, inverted_image, image2,
            and_image, or_image, xor_image, horizontal_flip, vertical_flip, perspective_image]

for i in range(len(images)):
    if len(images[i].shape) == 2:
        axes[i].imshow(images[i], cmap='gray')
    else:
        axes[i].imshow(images[i])
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()
