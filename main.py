import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from skimage.measure import find_contours
from skimage.morphology import binary_dilation
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import easyocr

im_1_path = './datasets/000/a01-000u.png'


def recognize_text(img_path):
    reader = easyocr.Reader(['en'])
    return reader.readtext(img_path, detail=0, paragraph=True)


# result = recognize_text(im_1_path)
# print(result)

img = cv2.imread(im_1_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
# digits = datasets.load_digits()
#
# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     ax.set_title('Training: %i' % label)

a = np.zeros((10, 12))
a[2, 8] = 1
a[3, 2] = 1
a[5, 6] = 1
a[3, 2] = 1
a[4, 7] = 1
a[9, 5] = 1
print(a)
# f = find_contours(a, 0.5)
# print(f)

kernel = np.ones((1, 3))  # array 1 x num
img_copy = binary_dilation(a, kernel)  # array of bools

print("img", img_copy)
bounding_boxes = find_contours(img_copy, 0.8)
print(bounding_boxes)

print()