#!/usr/bin/env python

import os, sys, shutil, glob
from bs4 import BeautifulSoup


object = sys.argv[1]

data = open(object,encoding='ISO-8859-1').read()

soup = BeautifulSoup(data, "html.parser")
page_data = soup.findAll('page')
print("Author:",soup.find('marcname').get_text())
print("Title:",soup.find('displaytitle').get_text())
print("Publication Date:",soup.find('pubdate').get_text())
print("Volume:",soup.find('currentvolume').get_text())
print("ESTCID:",soup.find('estcid').get_text())
print("found",len(page_data),"pages")
