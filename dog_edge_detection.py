import cv2
import matplotlib.pyplot as plt

image_path = 'messi.webp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

blur1 = cv2.GaussianBlur(image, (3, 3), 0)
blur2 = cv2.GaussianBlur(image, (9, 9), 0)

dog = cv2.absdiff(blur1, blur2)

dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

titles = ["Original Image", "Blurred (3x3)", "Blurred (9x9)", "Difference of Gaussian (DoG)"]
images = [image, blur1, blur2, dog]

for i in range(4):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(titles[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()
