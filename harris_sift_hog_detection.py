import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt

image = cv2.imread('house.jpg')
scene = cv2.imread('Saturn.bmp')
template = cv2.imread('saturn.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
scene_grey = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

def harris_corner_detection(img):
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=5, k=0.05)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [255, 0, 0] 
    return img

harris_result = harris_corner_detection(image.copy())

def sift_object_detection(scene, obj):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(obj, None)
    kp2, des2 = sift.detectAndCompute(scene, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    result_img = cv2.drawMatches(obj, kp1, scene, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return result_img

sift_result = sift_object_detection(scene_grey, gray_template)

def hog_feature_extraction(img):
    _, hog_img = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return hog_img

hog_result = hog_feature_extraction(gray_image)

fig, axs = plt.subplots(1, 3, figsize=(20, 8))

axs[0].imshow(cv2.cvtColor(harris_result, cv2.COLOR_BGR2RGB))
axs[0].set_title("Harris Corner Detection")

axs[1].imshow(sift_result, cmap='gray')
axs[1].set_title("SIFT Object Detection")

axs[2].imshow(hog_result, cmap='gray')
axs[2].set_title("HOG Feature Extraction")

plt.tight_layout()
plt.show()
