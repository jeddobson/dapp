#!/usr/bin/env python

import cv2
import imutils
import os, sys, shutil, glob
import numpy as np

import pickle

import nltk
from nltk.corpus import words

from bs4 import BeautifulSoup

object = sys.argv[1]

vocab = words.words()

# load and process pattern images
manicule_image = cv2.imread('share/manicule1.jpg')
manicule_gray = cv2.cvtColor(manicule_image, cv2.COLOR_BGR2GRAY)

annotation1_image = cv2.imread('share/annotation1.jpg')
annotation1_gray = cv2.cvtColor(annotation1_image, cv2.COLOR_BGR2GRAY)

annotation2_image = cv2.imread('share/annotation2.jpg')
annotation2_gray = cv2.cvtColor(annotation2_image, cv2.COLOR_BGR2GRAY)

annotation3_image = cv2.imread('share/annotation2.jpg')
annotation3_gray = cv2.cvtColor(annotation2_image, cv2.COLOR_BGR2GRAY)

asterisk_image = cv2.imread('share/asterisk1.jpg')
asterisk_gray = cv2.cvtColor(asterisk_image, cv2.COLOR_BGR2GRAY)

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

def find_annotation1(target,output):
    (tH, tW) = annotation1_gray.shape[:2]
    res = cv2.matchTemplate(target,annotation1_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.85
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(output, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_annotation2(target,output):
    (tH, tW) = annotation2_gray.shape[:2]
    res = cv2.matchTemplate(target,annotation2_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
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

    #print("ESTCID:",soup.find('estcid').get_text())
    #print("found",len(page_data),"pages")

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
        #print("page:",page_number,"paragraphs:",paragraph_count)

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
                #print("current:",position,content.strip(),"previous:",paragraph_matrix[len(paragraph_matrix) - 2][0])
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
        #text_space = text_space + ( (int(paragraph[0]) * (int(paragraph[1]))))
            text_space = text_space + ( (int(paragraph[2]) - int(paragraph[0])) * (int(paragraph[3]) - int(paragraph[1])))
    return(volume_dims,volume_text)


base="../LitAndLang_1/"


volume_dims,volume_text = page_reader(base + object + "/xml/" + object + ".xml")

processed_dir = base + object + "/processed"
image_dir = base + object + "/images/"

# check to see if the directory exists and remove
if os.path.isdir(processed_dir):
    shutil.rmtree(processed_dir)
os.mkdir(processed_dir)

found_objects=list()

idx=0 

for image_tif in glob.glob(image_dir + '*.TIF'):
    objects = 0
    page_dims = volume_dims[idx]
    page_image = cv2.imread(image_tif)
    gray_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    edges = cv2.Canny(gray_image,50,100)
   
    asterisk_c = find_asterisk(gray_image,page_image) 
    if asterisk_c > 0:
       objects = objects + 1
    manicule_c = find_manicule(gray_image,page_image)
    if manicule_c > 0:
       objects = objects + 1
    annotation1_c = find_annotation1(gray_image,page_image)
    if annotation1_c > 0:
       objects = objects + 1
    annotation2_c = find_annotation2(gray_image,page_image)
    if annotation2_c > 0:
       objects = objects + 1
    
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
        if word_c > 2:    
            cv2.rectangle(mask, (paragraph[0], paragraph[1]), (paragraph[2], paragraph[3]), 0, -1)
        cv2.rectangle(page_image, (paragraph[0], paragraph[1]), (paragraph[2], paragraph[3]), (0,255,0), 2)
    idx = pidx + 1

    edges = cv2.bitwise_and(edges, edges, mask=mask)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(edges,kernel,iterations = 1)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    
    pg_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    image_c = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
                    
        # if our object covers great than 2% of the page area
        if w * h > ( page_area * .02):
            #print(image_tif,x,y, x+w, y+h)
            cv2.rectangle(page_image,(x,y),(x+w,y+h),(255,0,0), 2)
            image_c = image_c +1
    objects = objects + image_c
            
    if objects > 0:
        output_file = processed_dir + "/" + os.path.basename(image_tif)
        cv2.imwrite(output_file,page_image)
        found_objects.append([image_tif,image_c,asterisk_c,manicule_c,annotation1_c,annotation2_c])
        output_pickle=open(processed_dir + '/objects.pkl','wb')
        pickle.dump(found_objects,output_pickle)
idx = idx + 1

