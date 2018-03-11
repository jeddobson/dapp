#!/usr/bin/env python
#
# James E. Dobson
# Dartmouth College
# jed@uchicago.edu
# 

# This script finds, catalogs, and draws borders around 
# interesting paratextual objects and ornaments in page
# images. It is used for the extraction of objects from 
# ECCO page images of eighteenth-century texts.

import cv2
import imutils
import os, sys, shutil, glob
import numpy as np

import pickle

import nltk
from nltk.corpus import words

from bs4 import BeautifulSoup

import argparse

# set default options
ocrboundaries = False
single_image = False

# parse arguments
parser = argparse.ArgumentParser(
    description='locates objects and annotates ECCO TIF documents')
parser.add_argument('object')
parser.add_argument('--draw-ocr-boundaries',help='place green boxes around paragraphs', dest='ocrboundaries', action='store_true')
parser.add_argument('--single-image', help='mark-up just a single page image', dest='single_image', action='store')
args = parser.parse_args()

object = args.object

if args.ocrboundaries == True:
   ocrboundaries = True

if args.single_image != False:
   single_image_switch = True
   single_image = args.single_image

if object == None:
   print("Error: need ECCO object")
   exit()

# load English language vocabulary
vocab = words.words()

################################################################################
# pre-load and process all images used for pattern matching
# convert to grayscale on load
################################################################################

manicule_gray = cv2.imread('share/manicule1.jpg',0)
arabesque_gray = cv2.imread('share/arabesque.jpg',0)
rosette_gray = cv2.imread('share/rosette.png',0)
annotation3_gray = cv2.imread('share/annotation3.jpg',0)
longdash_gray = cv2.imread('share/longdash.jpg',0)

# asterisms
asterism_gray = cv2.imread('share/asterism.jpg',0)
inverted_asterism_gray = cv2.imread('share/inverted_asterism.jpg',0)
asterism_block_gray = cv2.imread('share/asterism_block.jpg',0)
astrism_line_gray = cv2.imread('share/asterism_line.jpg',0)

#asterisk_image = cv2.imread('share/asterisk1.jpg')
#asterisk_gray = cv2.cvtColor(asterisk_image, cv2.COLOR_BGR2GRAY)

def find_inverted_asterism(target,output):
    (tH, tW) = inverted_asterism_gray.shape[:2]
    res = cv2.matchTemplate(target,inverted_asterism_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_asterism(target,output):
    (tH, tW) = asterism_gray.shape[:2]
    res = cv2.matchTemplate(target,asterism_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_asterism_line(target,output):
    (tH, tW) = asterism_line_gray.shape[:2]
    res = cv2.matchTemplate(target,asterism_line_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_asterism_block(target,output):
    (tH, tW) = asterism_block_gray.shape[:2]
    res = cv2.matchTemplate(target,asterism_block_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_longdash(target,output):
    (tH, tW) = longdash_gray.shape[:2]
    res = cv2.matchTemplate(target,longdash_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_manicule(target,output):
    (tH, tW) = manicule_gray.shape[:2]
    res = cv2.matchTemplate(target,manicule_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_arabesque(target,output):
    (tH, tW) = arabesque_gray.shape[:2]
    res = cv2.matchTemplate(target,arabesque_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.60
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_rosette(target,output):
    (tH, tW) = rosette_gray.shape[:2]
    res = cv2.matchTemplate(target,rosette_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.65
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_asterisk(target,output):
    (tH, tW) = asterisk_gray.shape[:2]
    res = cv2.matchTemplate(target,asterisk_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.70
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def page_reader(volume):
    file_name = volume
    data = open(file_name,encoding='ISO-8859-1').read()
    
    soup = BeautifulSoup(data, "html.parser")
    page_data = soup.findAll('page')

    ###############################################
    # calculate page size
    # present method is to find maximum x and y
    ###############################################

    page_size_max = soup.findAll('wd')
    max_length = max([int(word.get('pos').split(',')[:2][1]) for word in page_size_max])
    max_width = max([int(word.get('pos').split(',')[:2][0]) for word in page_size_max])
    page_size = max_length * max_width

    # start parsing each page
    volume_text = list()
    line_starting_position = list()

    volume_dataset=list()
    volume_dims=list()
    for page in page_data:
        page_line_starting_position=list()
        page_number = page.find('pageid').get_text()

        # get page dimensions
        word_matrix=list()
        t = page.findAll('wd')
        for x in t:
            word_matrix.append(x.get('pos'))

        paragraph_data = page.findAll('p')
        paragraph_count = len(paragraph_data)

        page_text=list()
        page_dims=list()

        for paragraph in paragraph_data:
            paragraph_matrix=list()
            words = paragraph.findAll('wd')
            pmin_x1 = min([int(w.get('pos').split(',')[:2][0]) for w in words])
            pmin_y1 = min([int(w.get('pos').split(',')[:2][1]) for w in words])
            pmax_x2 = max([int(w.get('pos').split(',')[2:][0]) for w in words])
            pmax_y2 = max([int(w.get('pos').split(',')[2:][1]) for w in words])
        
            # add x,y of first and last word
            #wordloc1 = words[0].get('pos').split(',')[:2]
            #wordloc2 = words[(len(words) - 1)].get('pos').split(',')[:2]
            #page_dims.append([wordloc1,wordloc2])
            page_dims.append([pmin_x1,pmin_y1,pmax_x2,pmax_y2])

            structured_list=list()

            temp_line=str()
            for word in words:
                lines=list()
                content = word.get_text()
                position = word.get('pos').split(',')
                
                paragraph_matrix.append(position)
                temp_line=temp_line + ' ' + content.strip()
                if int(position[0]) < int(paragraph_matrix[len(paragraph_matrix) - 2][0]):
                    page_line_starting_position.append(int(position[0]))
                    lines.append(temp_line)
                    temp_line=str()
                if(len(lines)) > 0:
                    structured_list.append(lines)
                page_text.append(structured_list)
        line_starting_position.append(page_line_starting_position)
        volume_dims.append(page_dims)
        volume_text.append(page_text)

        c=list()
        for y in page_text:
            if len(y) > 0:
                for t in y:
                    c.append(len(str(t[0]).split()))
    
        white_space=0
        text_space=0
        prior_y = 0
        prior_x = 0
    
        for paragraph in page_dims:
            text_space = text_space + ( (int(paragraph[2]) - int(paragraph[0])) * (int(paragraph[3]) - int(paragraph[1])))
    return(volume_dims,volume_text)


base="../LitAndLang_1/"

volume_dims,volume_text = page_reader(base + object + "/xml/" + object + ".xml")

processed_dir = base + object + "/processed"
image_dir = base + object + "/images/"

# check to see if the directory exists and remove
if os.path.isdir(processed_dir):
    shutil.rmtree(processed_dir,ignore_errors=True)
os.mkdir(processed_dir)

found_objects=list()

idx=0 

for image_tif in glob.glob(image_dir + '*.TIF'):

    # feature to run just on a single image (needs basename)
    if single_image_switch == True:
         if os.path.basename(image_tif) == single_image:
            pass
         else:
            continue

    page_dims = volume_dims[idx]

    # need to preserve color information 
    page_image = cv2.imread(image_tif)
    gray_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    edges = cv2.Canny(gray_image,50,100)
   
    #asterisk_c = find_asterisk(gray_image,page_image) 

    asterism_c = find_asterism(gray_image,page_image) 
    inverted_asterism_c = find_inverted_asterism(gray_image,page_image) 

    manicule_c = find_manicule(gray_image,page_image)
    arabesque_c = find_arabesque(gray_image,page_image)
    rosette_c = find_rosette(gray_image,page_image)
    #longdash_c = find_longdash(gray_image,page_image)
    longdash_c = 0 
    
    x, y = page_image.shape[:2] 
    page_area = x * y
    
    mask = np.ones(page_image.shape[:2], dtype="uint8") * 255

    pidx=0
    for paragraph in page_dims:
        word_c = 0
        sentences = volume_text[idx][pidx]
        if len(sentences) > 0:
            for s in sentences:
               tokens = nltk.word_tokenize(str(s))
               tokens_in_vocab = [word for word in tokens if word.lower() in vocab]
               word_c = word_c + len(tokens_in_vocab)

        # dump suspect OCR: only mask if we find more than two known words
        if word_c > 2:    
            cv2.rectangle(mask, (paragraph[0], paragraph[1]), (paragraph[2], paragraph[3]), 0, -1)

            # draw boundaries around paragraphs if requested
            if ocrboundaries == True:
               cv2.rectangle(page_image, (paragraph[0], paragraph[1]), (paragraph[2], paragraph[3]), (0,255,0), 2)

        pidx = pidx + 1

    # smooth page image and mask
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)

    #output_file = processed_dir + "/d" + os.path.basename(image_tif)
    #cv2.imwrite(output_file,dilation)

    # search for countours within dilated image
    pg_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    image_c = 0
    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)
                    
        # if area of our object is more than 20k pixels
        # also remove narrow (bars) or long objects (artifacts)
        image_max = 20000

        if w * h > image_max and w > (h / 8) and h > (w / 8):
            print("found object:",x,y,w,h)
        #if w * h > ( page_area * .05) and w > (h / 8):
            cv2.rectangle(page_image,(x,y),(x+w,y+h),(255,0,0), 2)
            image_c = image_c +1

    output_file = processed_dir + "/" + os.path.basename(image_tif)
    cv2.imwrite(output_file,page_image)

    # store list of found objects
    found_objects.append([image_tif,image_c,asterism_c,inverted_asterism_c,manicule_c,
        arabesque_c,rosette_c,longdash_c])
    idx = idx + 1

output_pickle=open(processed_dir + '/objects_' + object + '.pkl','wb')
pickle.dump(found_objects,output_pickle)
