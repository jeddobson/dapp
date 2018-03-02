#!/usr/bin/env python

import pickle

data = pickle.load(open(sys.argv[1]),'rb')

for x in data:
	if sum(x[1:5]) != 0:
		print(x)
