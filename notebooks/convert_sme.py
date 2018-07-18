
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def load_file_data(path):
    file = open(path, 'r')
    data = []
    for line in file:
        data.append(line.strip("\n").split())
    return data


# In[5]:


mapping = dict(load_file_data('data/fb15k/id2entity.sme.txt'))


# In[32]:


elements = np.empty(14952, dtype='object')
for k in mapping:
    if (int(k) > 14951):
        continue
    elements[int(k)] = mapping[k]


# In[ ]:


file = open('data/fb15k/predictions.transe.txt', 'r')
out = open('data/fb15k/predictions.sme.inverse.txt', 'w')
n = 300

for line in file:
    data = line.strip("\n").split(" ")
    scores = zip(elements[np.argsort(data[3:])][:n], np.sort(data[3:])[:n])
    joined = []
    for a, b in scores:
        if a == elements[14951]:
            continue
        joined.append(a + " " + b)
    
    out.write(" ".join(joined))
    out.write("\n")
        
file.close()
out.close()


