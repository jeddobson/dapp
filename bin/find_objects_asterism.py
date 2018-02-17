#!/usr/bin/env python

import cv2
import imutils
import os, sys, shutil, glob
import numpy as np

import pickle

object = sys.argv[1]

# load and process pattern images
asterism_gray = cv2.imread('share/asterism.jpg',0)

def find_asterism(target,output):
    (tH, tW) = asterism_gray.shape[:2]
    res = cv2.matchTemplate(target,asterism_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        #cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

base="../LitAndLang_1/"

image_dir = base + object + "/images/"
output_dir = base + object + "/"

found_objects=list()

for image_tif in glob.glob(image_dir + '*.TIF'):
    objects = 0
    gray_image = cv2.imread(image_tif, 0)
    asterism_c = find_asterism(gray_image,gray_image) 

    found_objects.append([image_tif,asterism_c])

output_pickle=open(output_dir + '/asterism_count.pkl','wb')
pickle.dump(found_objects,output_pickle)

