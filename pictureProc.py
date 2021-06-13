import os
import random
import time
from os import walk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.feature import local_binary_pattern
from skimage.measure import find_contours
from skimage.morphology import binary_dilation
from sklearn.svm import SVC
from torch import nn, optim
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import pytesseract

import easyocr

OVERLAPPING_METHOD = 0
LINES_METHOD = 1
torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_images(images, titles=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def preprocess_image(img, feature_extraction_method):
    if feature_extraction_method == OVERLAPPING_METHOD:
        img_copy = img.copy()

        if len(img.shape) > 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)  # conversion to the other color space(gray)
        img_copy = cv2.medianBlur(img_copy, 5)  # (reduce noise) non-linear technique takes a median of all the pixels under the kernel area and replaces the central element with this median value.
        show_images([img, img_copy], ["img", 'blur'])
        img_copy = cv2.threshold(img_copy, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # progowanie ale trza sprawdzić jakie konkretnie
        show_images([img, img_copy])
        min_vertical, max_vertical = get_corpus_boundaries(img_copy)
        label_image = img_copy[:min_vertical]
        reader = easyocr.Reader(['en'])
        labels = reader.readtext(label_image, detail=0, paragraph=True)
        label = labels[len(labels) - 1].split()
        print("----------------------------------------------")
        print(label)
        print("----------------------------------------------")
        img_copy = img_copy[min_vertical:max_vertical]  # cut edges of image to handwriten part
        show_images([label_image, img_copy], ['labels', 'scope'])
        # print("imgshape", img_copy.shape)east
        return img_copy, label
    if feature_extraction_method == LINES_METHOD:
        img_copy = img.copy()
        if len(img.shape) > 2:
            grayscale_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_img = img.copy()
        img_copy = cv2.threshold(grayscale_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        min_vertical, max_vertical = get_corpus_boundaries(img_copy)
        label_image = img_copy[:min_vertical]
        reader = easyocr.Reader(['en'])
        labels = reader.readtext(label_image, detail=0, paragraph=True)
        label = labels[len(labels) - 1].split()
        print("----------------------------------------------")
        print(label)
        print("----------------------------------------------")
        img_copy = img_copy[min_vertical:max_vertical]
        grayscale_img = grayscale_img[min_vertical:max_vertical]
        filter_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_copy_sharpened = cv2.filter2D(img_copy, -1, filter_kernel)
        return img_copy_sharpened, grayscale_img


def get_corpus_boundaries(img):
    crop = []
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))  # it mark places where horizontal lines
    show_images([img, horizontal_kernel], ['img', 'horiz'])
    detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)  # Opening is just another name of erosion followed by dilation. It is useful in removing noise.
    show_images([img, detect_horizontal], ['img', 'detect'])
    contours = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)  # marked contours of edges| returns 2 or 3 values
    # show_images([img, contours],['img','cont'])
    contours = contours[0] if len(contours) == 2 else contours[1]
    prev = -1

    for i, c in enumerate(contours):
        if np.abs(prev - int(c[0][0][1])) > 800 or prev == -1:
            crop.append(int(c[0][0][1]))
            prev = int(c[0][0][1])
    # print(crop)
    crop.sort()
    max_vertical = crop[1] - 20
    min_vertical = crop[0] + 20
    return min_vertical, max_vertical


def segment_image(img, num, grayscale_img=None):  # image,3
    if grayscale_img is not None:
        grayscale_images = []
        img_copy = np.copy(img)
        kernel = np.ones((1, num))
        img_copy = binary_dilation(img_copy, kernel)
        show_images([img, img_copy], ['img', 'binary_dilation'])
        bounding_boxes = find_contours(img_copy, 0.8)
        for box in bounding_boxes:
            x_min = int(np.min(box[:, 1]))
            x_max = int(np.max(box[:, 1]))
            y_min = int(np.min(box[:, 0]))
            y_max = int(np.max(box[:, 0]))
            if (y_max - y_min) > 50 and (x_max - x_min) > 50:
                grayscale_images.append(grayscale_img[y_min:y_max, x_min:x_max])

        # grayscale_images= find_contours(grayscale_images, 0.8)
        show_images(grayscale_images[:10])

        # for gray in grayscale_images:
        #     tmp = []
        #     ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        #     contours= cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #     for cnt in contours[1]:
        #         x, y, w, h = cv2.boundingRect(cnt)
        #         # bound the images
        #         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #     i = 0
        #     for cnt in contours:
        #         x, y, w, h = cv2.boundingRect(cnt)
        #
        #         if w > 50 and h > 50:
        #             # save individual images
        #             tmp.append(thresh1[y:y + h, x:x + w])
        #             i = i + 1
        #     # # img_copy=gray
        #     # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        #     # show_images([gray, gray], ["gray", 'blur'])
        #     # bounding_boxes = find_contours(gray, 0.5)
        #     # for box in bounding_boxes:  # extract individual letters or words
        #     #     x_min = int(np.min(box[:, 1]))
        #     #     x_max = int(np.max(box[:, 1]))
        #     #     y_min = int(np.min(box[:, 0]))
        #     #     y_max = int(np.max(box[:, 0]))
        #     #     if (y_max - y_min) > 10 and (x_max - x_min) > 10:
        #     #         tmp.append(gray[y_min:y_max, x_min:x_max])
        #     show_images(tmp[:10])
        return grayscale_images
    else:
        images = []
        img_copy = np.copy(img)
        kernel = np.ones((1, num))  # array 1 x num
        # print("img", img_copy)
        img_copy = binary_dilation(img_copy, kernel)  # array of bools
        show_images([img, img_copy], ['img', 'binary_dilation'])
        # print("img2", img_copy)
        bounding_boxes = find_contours(img_copy,
                                       0.8)  # , fully_connected='low' 0.8- Value along which to find contours in the array. trza doczytać jak dokładnie działa
        # show_images([img, bounding_boxes], ['img', 'bounding_boxes'])
        # print('box',bounding_boxes)
        # print('len', len(bounding_boxes))
        z = np.array(bounding_boxes[0])
        print('shape', z.shape)
        print("box", bounding_boxes[0])
        for box in bounding_boxes:  # extract individual letters or words
            x_min = int(np.min(box[:, 1]))
            x_max = int(np.max(box[:, 1]))
            y_min = int(np.min(box[:, 0]))
            y_max = int(np.max(box[:, 0]))
            if (y_max - y_min) > 10 and (x_max - x_min) > 10:
                images.append(img[y_min:y_max, x_min:x_max])
        show_images(images[:10])
        print('shape', images[0])
        print('shapex', images[0][1].shape)
        print('shapey', images[0].shape)
        print('shape[1]', images[0].shape[1])

        return images


def read(root):
    images = []  # images table
    test_images = []

    for i in range(3):
        found_images = False
        while not found_images:
            images_path = root
            random_writer = random.randrange(672)  # losowa liczba 0-672
            if random_writer < 10:
                random_writer = "00" + str(random_writer)  # dla folderów o numerze<10
            elif random_writer < 100:
                random_writer = "0" + str(random_writer)  # dla folderów o numerze<100
            images_path = os.path.join(images_path, str(random_writer))  # ścieżka do konkretnego folderu z danymi
            if not os.path.isdir(images_path):
                continue
            _, _, filenames = next(walk(images_path))  # następny plik w folderze
            if len(filenames) <= 2 and i == 2 and len(
                    test_images) == 0:  # jeśli liczba zdjęć w folderze jest <=2 i jest to 3 iteracja i zbiór na dane testowe jest pusty to pomiń
                continue
            if len(filenames) >= 2:  # dla folderów z wielomoa zdjęciami
                found_images = True  # kończymy pętlę while
                chosen_filenames = []
                for j in range(2):

                    random_filename = random.choice(filenames)  # wybieramy radnomowo zdjęcie z folderu
                    while random_filename in chosen_filenames:  # tylko dla 2 iteracji pomijamy duplikaty
                        random_filename = random.choice(filenames)
                    chosen_filenames.append(random_filename)  # lista wybranych zdjęć
                    images.append(cv2.imread(os.path.join(images_path, random_filename)))  # saved photos

                if len(filenames) >= 3:  # if we have more than 3 images make test data
                    random_filename = random.choice(filenames)
                    while random_filename in chosen_filenames:
                        random_filename = random.choice(filenames)
                    chosen_filenames.append(random_filename)
                    test_images.append(cv2.imread(os.path.join(images_path, random_filename)))
    test_choice = random.randint(0, len(test_images) - 1)
    test_image = test_images[test_choice]  # choice only one test image

    return images, test_image


def extract_features(images, feature_extraction_method):
    if feature_extraction_method == LINES_METHOD:
        lines_labels = []
        lines = []
        for image in images:
            image, grayscale_image = preprocess_image(image, feature_extraction_method)
            grayscale_lines = segment_image(image, 100, grayscale_image)
            for line in grayscale_lines:
                lines.append(line)
        return lines

    if feature_extraction_method == OVERLAPPING_METHOD:
        textures = []
        textures_labels = []
        for image in images:
            image, label = preprocess_image(image)  # only handwriten part
            words = segment_image(image, 3)  # images of words or letters
            avg_height = 0
            for word in words:
                avg_height += word.shape[0] / len(words)  # average height of letters
            print('averageheight', avg_height)

            return textures, textures_labels


epochs = 1
root = 'datasets'

for epoch in range(epochs):
    images, test_image, = read(root)  # choice learning set and test set
    # show_images(images)
    extract_features(images, LINES_METHOD)
