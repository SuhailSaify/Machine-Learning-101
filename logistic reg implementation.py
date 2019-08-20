#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.datasets  import make_blobs
import matplotlib.pyplot as plt


# In[14]:



plt.style.use("ggplot")


# In[46]:


X,Y = make_blobs(centers=2,n_samples=500,random_state=11,n_features=2)
plt.scatter(X[:,0],X[:,1],c=Y)
print(X.shape)
plt.show()


# In[24]:


def sigmoid(z):
 return (1.0)/(1+np.exp(-z))


# In[27]:


#z=np.array([1,2,4,5,6])
#z=sigmoid(z)
#print(z)


# In[63]:


def predict(X,weights):
    z=np.dot(X,weights)
    pred=sigmoid(z)
    return pred

def loss(X,Y,weights):
    Y_=predict(X,weights)
    cost = -np.mean((Y*np.log(Y_))+((1-Y)*(np.log(1-Y_))))
    return cost                

def update(X,Y,weights,learning_rate):
     Y_=predict(X,weights)
     dw=np.dot(X.T,Y_-Y)
    
     m=X.shape[0]
        
     weights=weights-learning_rate*dw / (float(m))
     return weights

def train(X,Y,learning_rate=0.5, maxEpochs=100):
 
    ones=np.ones((X.shape[0],1))
    X=np.hstack((ones,X))
    
    weights=np.zeros(X.shape[1])
    
    for epoch in range(maxEpochs):
        
        weights=update(X,Y,weights,learning_rate)
        
        if(epoch%10==0):
         l=loss(X,Y,weights)
         print("Epoch %d Loss %.4f"%(epoch,l))
    
    return weights


# In[90]:


train(X,Y,learning_rate=1.2,maxEpochs=200)


# In[ ]:




