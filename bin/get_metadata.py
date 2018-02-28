#!/usr/bin/env python

import os, sys, shutil, glob
from bs4 import BeautifulSoup
import argparse


parser = argparse.ArgumentParser(
       description='getmetadata: extract metadata from ECCO xml files')
parser.add_argument('--delimited',help='produce delimited output',dest='delimited',action='store_true')
parser.add_argument('filename')
args = parser.parse_args()

if os.path.exists(args.filename) == False:
   print("cannot open",args.filename)
   exit()

data = open(args.filename,encoding='ISO-8859-1').read()

soup = BeautifulSoup(data, "html.parser")

# authorname
try:
    author=soup.find('marcname').get_text()
except:
    author="None"

# title
try:
    title=soup.find('displaytitle').get_text()
except:
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


if args.delimited == True:
    print(author,'|',title,'|',estcid,'|',pubdate,'|',volume)
else:
    print("Author:",author)
    print("Title:",title)
    print("Publication Date:",pubdate)
    print("Volume:",volume)
    print("ESTCID:",estcid)
