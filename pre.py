import numpy as np


data = np.loadtxt("./data/hopperdata1.txt")

l = []

for i in range(100):
    for j in range (100):
        if(i!=j):
            if(data[i][-1] > data[j][-1]):
                d = np.concatenate((data[i][:-1],data[j][:-1],[1]))
            else:
                d = np.concatenate((data[i][:-1],data[j][:-1],[0]))
            l.append(d)

l = np.array(l)

print(l.shape)

np.savetxt("./data/prehopper.txt", l)