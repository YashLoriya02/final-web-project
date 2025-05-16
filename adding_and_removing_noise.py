import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    total_pixels = image.size

    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

def remove_gaussian_noise(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def remove_salt_and_pepper_noise(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def remove_noise_nlmeans(image, h=5):
    return cv2.fastNlMeansDenoising(image, None, h, 7, 21)

image = cv2.imread('messi.webp', cv2.IMREAD_GRAYSCALE)

gaussian_noisy = add_gaussian_noise(image)
gaussian_denoised = remove_gaussian_noise(gaussian_noisy)
gaussian_nlmeans = remove_noise_nlmeans(gaussian_denoised)

sp_noisy = add_salt_and_pepper_noise(image)
sp_denoised = remove_salt_and_pepper_noise(sp_noisy)
sp_nlmeans = remove_noise_nlmeans(sp_denoised)

gaussian_noisy = add_gaussian_noise(image)
combined_noisy = add_salt_and_pepper_noise(gaussian_noisy)
denoised_image = remove_noise_nlmeans(combined_noisy)

fig, axs = plt.subplots(3, 4, figsize=(12, 16))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title("Original Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(gaussian_noisy, cmap='gray')
axs[0, 1].set_title("Gaussian Noise")
axs[0, 1].axis('off')

axs[0, 2].imshow(gaussian_denoised, cmap='gray')
axs[0, 2].set_title("Gaussian Denoised (Blur)")
axs[0, 2].axis('off')

axs[0, 3].imshow(gaussian_nlmeans, cmap='gray')
axs[0, 3].set_title("Gaussian Denoised (NL Means)")
axs[0, 3].axis('off')

axs[1, 0].imshow(image, cmap='gray')
axs[1, 0].set_title("Original Image")
axs[1, 0].axis('off')

axs[1, 1].imshow(sp_noisy, cmap='gray')
axs[1, 1].set_title("Salt & Pepper Noise")
axs[1, 1].axis('off')

axs[1, 2].imshow(sp_denoised, cmap='gray')
axs[1, 2].set_title("Salt & Pepper Denoised (Median)")
axs[1, 2].axis('off')

axs[1, 3].imshow(sp_nlmeans, cmap='gray')
axs[1, 3].set_title("Salt & Pepper Denoised (NL Means)")
axs[1, 3].axis('off')

axs[2, 0].imshow(image, cmap='gray')
axs[2, 0].set_title("Original Image")
axs[2, 0].axis('off')

axs[2, 1].imshow(gaussian_noisy, cmap='gray')
axs[2, 1].set_title("Gaussian Noise")
axs[2, 1].axis('off')

axs[2, 2].imshow(combined_noisy, cmap='gray')
axs[2, 2].set_title("Gaussian + Salt & Pepper Noise")
axs[2, 2].axis('off')

axs[2, 3].imshow(denoised_image, cmap='gray')
axs[2, 3].set_title("Denoised (NL Means)")
axs[2, 3].axis('off')

plt.show()
