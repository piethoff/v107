#Use:
#python var.py [data].txt [column]
#

import numpy as np
import sys

def mean(a):
    sum = 0
    for i in a:
        sum += i
    sum = sum/len(a)
    return sum

def var(a):
    mean2 = mean(a)
    sum = 0
    for i in a:
        sum += (i-mean2)**2
    sum = sum/len(a)
    sum = sum/(len(a) - 1)
    return np.sqrt(sum)

data = np.genfromtxt(str(sys.argv[1]), unpack=True)

#print(data[int(sys.argv[2])])
print(mean(data[int(sys.argv[2])]), "+/-", var(data[int(sys.argv[2])]), sep="")
