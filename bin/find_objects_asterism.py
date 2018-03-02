#!/usr/bin/env python

import cv2
import imutils
import os, sys, shutil, glob
import numpy as np

import pickle

object = sys.argv[1]

# load and process pattern images
# asterisms
asterism_gray = cv2.imread('share/asterism.jpg',0)
inverted_asterism_gray = cv2.imread('share/inverted_asterism.jpg',0)
asterism_block_gray = cv2.imread('share/asterism_block.jpg',0)
astrism_line_gray = cv2.imread('share/asterism_line.jpg',0)

def find_inverted_asterism(target,output):
    (tH, tW) = inverted_asterism_gray.shape[:2]
    res = cv2.matchTemplate(target,inverted_asterism_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
    return(count)

def find_asterism(target,output):
    (tH, tW) = asterism_gray.shape[:2]
    res = cv2.matchTemplate(target,asterism_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
    return(count)

def find_asterism_line(target,output):
    (tH, tW) = asterism_line_gray.shape[:2]
    res = cv2.matchTemplate(target,asterism_line_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
    return(count)

def find_asterism_block(target,output):
    (tH, tW) = asterism_block_gray.shape[:2]
    res = cv2.matchTemplate(target,asterism_block_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
    return(count)

base="../LitAndLang_1/"

image_dir = base + object + "/images/"
output_dir = base + object + "/"

found_objects=list()

for image_tif in glob.glob(image_dir + '*.TIF'):
    objects = 0
    gray_image = cv2.imread(image_tif, 0)

    asterism_c = find_asterism(gray_image,gray_image) 
    asterism_block_c = find_asterism_block(gray_image,gray_image) 
    asterism_line_c = find_asterism_line(gray_image,gray_image) 
    inverted_asterism_c = find_inverted_asterism(gray_image,gray_image) 

    found_objects.append([image_tif,asterism_c,inverted_asterism_c,asterism_block_c,asterism_line_c])

output_pickle=open(output_dir + '/asterism_count.pkl','wb')
pickle.dump(found_objects,output_pickle)
