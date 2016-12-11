import numpy as np
import re

import matplotlib.pyplot as plt

with open("logging_data2.txt",'r') as f:
    lines = f.readlines()
lines = lines[5:]

K, V = [], []

validation_costs = []
for i, l in enumerate(lines):
    print(i)
    ll = l.split(',')
    firstword = ll[0].split(' ')[0]
    if firstword == 'Epoch':
        vc = float(re.findall("\d+\.\d+",l)[0])
        validation_costs.append(vc)
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
V = np.array(V).astype('float32')
V = np.reshape(V,(30,6,7))
V = np.mean(V,axis=1)

#f, axs = plt.subplots(1,len(keys))
#for i, ax in enumerate(axs):
    #ax.plot(V[:,i])
    #ax.set_title(keys[i])
    
f, ax = plt.subplots(1,1)
ks = ['_cost']
names = ["Cost"]

d = dict(zip(ks,names))
#for k, ax in zip(ks,axs):
ax.set_xlim([1,30])
rt = range(1,31)
i = keys.index(ks[0])
ht = ax.plot(rt,V[:,i])
hv = ax.plot(rt,validation_costs,color='red')
ax.set_title(d[ks[0]])

ax.legend(handles=ht+hv,labels=['Train','Test'])
    
plt.show()
halt= True