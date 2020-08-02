#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pakcages
import sys
import numpy as np 
import pandas as pd
import nltk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter(action='ignore')

get_ipython().system('{sys.executable} -m pip install -U textblob')

get_ipython().run_line_magic('run', './Text_Normalization_Function.ipynb')


# In[2]:


# Read in data
df_data=pd.read_csv('data.csv')


# # 1. Perprocessing

# ### 1) Split train and test dataset

# In[116]:


df_train=df_H.sample(frac=0.7, random_state=7)
df_test=df_H.drop(df_train.index)


# In[ ]:


# Split dependent variable and independent variables
# Y
Y=df_train[y]
Y_test=df_test[y]
# X
X_train=df_train[feature]
X_test=df_test[feature]


# ### 2) Scale (using standard scaling)

# In[119]:


# Scale independent variables
scaler = preprocessing.StandardScaler().fit(X_train)
X=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# ## 2. KNN Model

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error 
from math import sqrt


# In[ ]:


def best_KNN(X,Y,X_test,Y_test,n=15):
    
    # Fit KNN model with train dataset and try different Ks
    mse_val = [] 
    for K in range(n):
        K = K+1
        model = neighbors.KNeighborsRegressor(n_neighbors = K) # set k value
        model.fit(X, Y)# fit knn model   
        pred=model.predict(X_test)# make prediction
        error = sum(abs(Y_test-pred))/sum(Y_test) # calculate WAPE
        mse_val.append(error) 
        print('WAPE value for k= ' , K , 'is:', error)
    
    # Plotting the rmse values against k values
    curve = pd.DataFrame(mse_val)

    curve.plot()
    plt.xlabel("Number of nearest neighborhoods")
    plt.ylabel("WAPE")
    plt.title("Accuracy Rate against k values")
    plt.show()
    
    # Print best k and minium error
    print('The best k is ', mse_val.index(min(mse_val))+1)
    print('The minium WAPE is ',min(mse_val))


# ## 3. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# ### Hyperparameter

# In[12]:


# choose the best number of estimators
rf_list=[]
for rf_estimator in range(10, 50,1):
    rf=RandomForestRegressor(n_estimators=rf_estimator)
    rf.fit(x_train,y_train)
    y_hat=rf.predict(x_test)
    accuracy = sum(abs(y_hat-y_test))/sum(y_test)
    rf_list.append(accuracy)

index=rf_list.index(min(rf_list))
grif=range(10, 50,1)[index]
print(grif)
min(rf_list)


# ## 4. XGboost

# In[ ]:


from xgboost import XGBRegressor


# ### Hyperparameter

# In[18]:


# choose the best learning rate
k_list=[]
for kkk in np.linspace(0.05, 0.30, num=25):
    xg=XGBRegressor(learning_rate=kkk,n_estimators=100,min_child_weight=2,max_depth=7)
    xg.fit(x_train,y_train)
    y_hat=xg.predict(x_test)
    accuracy = sum(abs(y_hat-y_test))/sum(y_test)
    k_list.append(accuracy)


# In[19]:


index=k_list.index(min(k_list))
grif=np.linspace(0.05, 0.30, num=25)[index]
print(grif)


# In[20]:


# choose the best number of estimators
n_list=[]
for kkk in range(100, 1000,10):
    xg=XGBRegressor(learning_rate=grif,n_estimators=kkk,min_child_weight=2,max_depth=7)
    xg.fit(x_train,y_train)
    y_hat=xg.predict(x_test)
    accuracy = sum(abs(y_hat-y_test))/sum(y_test)
    n_list.append(accuracy)


# In[21]:


index=n_list.index(min(n_list))
grif2=range(100, 1000,10)[index]
print(grif2)
print(min(n_list))


# ## 5. Neural Network

# ### 5.1 Hyperparameter

# In[26]:


# choose the best hidden layer
for kkkk in range(50,550,10):
    dnn_list=[]
    mlp=MLPRegressor(hidden_layer_sizes = (kkkk,kkkk),
                 activation = 'relu',\
                 solver = 'adam',\
                 alpha = 0.0001,\
                 batch_size = 'auto',\
                 learning_rate = 'constant',\
                 learning_rate_init = 0.006,
                 power_t = 0.5,\
                 max_iter = 1000,\
                 shuffle = True,\
                 random_state = 42,\
                 tol = 0.0001,\
                 verbose = False,\
                 warm_start = False,\
                 momentum = 0.9,\
                 nesterovs_momentum = True,\
                 early_stopping = False,\
                 validation_fraction = 0.1,\
                 beta_1 = 0.9,\
                 beta_2 = 0.999,\
                 epsilon = 1e-08)
    mlp.fit(x_train,y_train)
    y_hat=mlp.predict(x_test)
    accuracy = sum(abs(y_hat-y_test))/sum(y_test)
    dnn_list.append(accuracy)


# In[27]:


index=dnn_list.index(min(dnn_list))
grif=range(50,550,10)[index]
print(grif)
print(min(dnn_list))


# In[28]:


# choose the best learning rate
for kkk in np.linspace(0.001,0.01,num=20):
    dnn_list_2=[]
    mlp=MLPRegressor(hidden_layer_sizes = (grif,grif),
                 activation = 'relu',\
                 solver = 'adam',\
                 alpha = 0.0001,\
                 batch_size = 'auto',\
                 learning_rate = 'constant',\
                 learning_rate_init = kkk,
                 power_t = 0.5,\
                 max_iter = 1000,\
                 shuffle = True,\
                 random_state = 42,\
                 tol = 0.0001,\
                 verbose = False,\
                 warm_start = False,\
                 momentum = 0.9,\
                 nesterovs_momentum = True,\
                 early_stopping = False,\
                 validation_fraction = 0.1,\
                 beta_1 = 0.9,\
                 beta_2 = 0.999,\
                 epsilon = 1e-08)
    mlp.fit(x_train,y_train)
    y_hat=mlp.predict(x_test)
    accuracy = sum(abs(y_hat-y_test))/sum(y_test)
    dnn_list_2.append(accuracy)


# In[29]:


index=dnn_list_2.index(min(dnn_list_2))
grif2=np.linspace(0.001,0.01,num=20)[index]
print(grif2)
print(min(dnn_list_2))


# ### 5.2 Fit the model

# In[34]:


mlp=MLPRegressor(hidden_layer_sizes = (grif,grif),
                 activation = 'relu',\
                 solver = 'adam',\
                 alpha = 0.0001,\
                 batch_size = 'auto',\
                 learning_rate = 'constant',\
                 learning_rate_init = grif2,
                 power_t = 0.5,\
                 max_iter = 1000,\
                 shuffle = True,\
                 random_state = 42,\
                 tol = 0.0001,\
                 verbose = False,\
                 warm_start = False,\
                 momentum = 0.9,\
                 nesterovs_momentum = True,\
                 early_stopping = False,\
                 validation_fraction = 0.1,\
                 beta_1 = 0.9,\
                 beta_2 = 0.999,\
                 epsilon = 1e-08)
mlp.fit(x_train,y_train)
y_hat=mlp.predict(x_test)
print(sum(abs(y_hat-y_test))/sum(y_test))


# In[31]:


y_hat=mlp.predict(x_test)


# ### 5.3 Output result

# In[35]:


dnn_df = pd.DataFrame({'y_hat':y_hat,'y_test':y_test})
diff = list((abs(y_hat-y_test)))
dnn_df['diff'] = diff
dnn_df['SKU Code']=x_test['SKU Code']


# In[ ]:




