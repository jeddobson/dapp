#!/usr/bin/env python

import os, sys, shutil, glob
from bs4 import BeautifulSoup


object = sys.argv[1]

data = open(object,encoding='ISO-8859-1').read()

soup = BeautifulSoup(data, "html.parser")

# authorname
try:
    author=soup.find('marcname').get_text()
execept:
    author="None"

# title
try:
    title=soup.find('displaytitle').get_text()
execept:
    title="None"

# publication date
try:
    pubdate=soup.find('pubdate').get_text()
except:
    pubdate="None" 

# volume number
try:
    volume=soup.find('currentvolume').get_text()
except:
    volume="None" 

# ESTCID 
try:
    estcid=soup.find('estcid').get_text()
except:
    estcid="None" 


print("Author:",author)
print("Title:",title)
print("Publication Date:",pubdate)
print("Volume:",volume)
print("ESTCID:",estcid)
