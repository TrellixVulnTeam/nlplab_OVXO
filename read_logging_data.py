import numpy as np
import re

import matplotlib.pyplot as plt

with open("logging_data2.txt",'r') as f:
    lines = f.readlines()
lines = lines[5:]

K, V = [], []
for i, l in enumerate(lines):
    print(i)
    ll = l.split(',')
    firstword = ll[0].split(' ')[0]
    if firstword == 'Epoch':
        continue
    
    if firstword != 'epoch':
        break
    keys = []
    vals = []
    for lll in ll[:-1]: # ignoring last key, which contains two values ...
        llll = lll.split(' ')
        key = '_'.join(llll[:-1])
        val = llll[-1]
        
        keys.append(key)
        vals.append(val)
    V.append(np.array(vals))
    
# matrix containing all the values we care about.
V = np.array(V)

f, axs = plt.subplots(1,len(keys))
for i, ax in enumerate(axs):
    ax.plot(V[:,i])
    ax.set_title(keys[i])
    
plt.show()
halt= True