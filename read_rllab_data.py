import numpy as np
import re

import matplotlib.pyplot as plt

with open("rllab_ac_12_10.txt",'r') as f:
    lines = f.readlines()
lines = lines[5:]

K, V = [], []
avg_returns = []
avg_env_returns = []
for i, l in enumerate(lines):
    floats = re.findall("\d+\.\d+",l)
    val = floats[-1]
    
    if 'AverageReturn' in l:
        avg_returns.append(val)
    if 'AverageEnvReturn' in l:
        avg_env_returns.append(val)
    
V = np.array(V)

rt = range(1,len(avg_env_returns)+1)
    
avg_returns = np.array(avg_returns).astype('float32')
avg_env_returns = np.array(avg_env_returns).astype('float32')
f, axs = plt.subplots(1,2)

axs[0].set_xlim(1,len(avg_env_returns)+1)
axs[0].plot(rt,avg_returns)
axs[0].set_title("Avg. Return from Critic")

axs[1].set_xlim(1,len(avg_env_returns)+1)
axs[1].plot(rt,avg_env_returns)
axs[1].set_title("Avg. Return from Environment")

plt.show()
halt= True