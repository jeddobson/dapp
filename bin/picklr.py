#!/usr/bin/env python

# trivial script to display any possible asterisms found

import pickle, sys

data = pickle.load(open(sys.argv[1]),'rb')

for x in data:
 if len(x) > 4:
   if sum(x[1:5]) != 0:
     print(x)
