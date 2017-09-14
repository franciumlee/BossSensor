# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2

IMAGE_SIZE = 64

def CaptureFace(InputCapture,OutputFile,face_number=0):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.VideoCapture(InputCapture)
    if (img.isOpened() == False):
        print("Error opening video stream or file")

    while(img.isOpened()!=False):
        ret,frame=img.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_color = frame[y:y + h, x:x + w]
            printNumber=face_number
            a=OutputFile+str(printNumber)+'boss1'+'.jpg'
            print a
            cv2.imwrite(a,roi_color)
            face_number=face_number+1
            print face_number


def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels


def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)

    labels = np.array([0 if label.endswith('1') | label.endswith('2') else 1 for label in labels])
    print labels
    return images, labels
