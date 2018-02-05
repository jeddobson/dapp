import cv2
import imutils
import numpy as np
import glob

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

def find_manicule(target):
    (tH, tW) = manicule_gray.shape[:2]
    res = cv2.matchTemplate(target,manicule_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.75
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(target, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)


def find_annotation1(target):
    (tH, tW) = annotation1_gray.shape[:2]
    res = cv2.matchTemplate(target,annotation1_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.85
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(target, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_annotation2(target):
    (tH, tW) = annotation2_gray.shape[:2]
    res = cv2.matchTemplate(target,annotation2_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.85
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(target, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)

def find_asterisk(target):
    (tH, tW) = asterisk_gray.shape[:2]
    res = cv2.matchTemplate(target,asterisk_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.85
    locations = np.where(res >= threshold)
    count=0
    for pt in zip(*locations[::-1]):
        count = count + 1
        cv2.rectangle(target, pt, (pt[0] + tW, pt[1] + tH), (0,0,255), 2)
    return(count)


image_dir = "ECCO/*/images/"
found_objects=list()
for image_tif in glob.glob(image_dir + '*.TIF'):
    
    page_image = cv2.imread(image_tif)
    gray_image = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    #gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    #edges = cv2.Canny(gray_image,100,200)    
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    #smoothed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
   
    found_objects.append([image_tif,find_asterisk(gray_image), find_manicule(gray_image),
                          find_annotation1(gray_image),find_annotation2(gray_image)])
    idx = idx + 1
