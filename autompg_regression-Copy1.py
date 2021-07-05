#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn


# In[2]:


print('a')


# In[3]:


from sklearn.preprocessing import StandardScaler
X = [[0, 15], [1, -10]]
StandardScaler().fit(X).transform(X)


# In[4]:


import pandas as pd


# In[5]:


pd_data = pd.read_csv('./files/auto-mpg.csv', header=None)
pd_data.info()


# In[6]:


pd_data.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']


# In[7]:


from sklearn.linear_model import LinearRegression


# In[22]:


x=pd_data[['weight','cylinders']]


# In[23]:


y=pd_data[['mpg']]



lr = LinearRegression()


lr.fit(x,y)


lr.coef_, lr.intercept_ #기울기, 절편

lr.score(x,y) #정확도

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y)


lr.fit(X_train, Y_train)

print(lr.score(X_train, Y_train))

print(lr.score(X_test,Y_test))


# In[35]:


lr.predict([[3504.0, 8]])


# In[36]:


lr.predict([[2790.0, 4]])


# In[38]:


import pickle


# In[40]:


pickle.dump(lr,open('./saves/autompg_lr.pkl','wb'))


# In[ ]:


abc = pickle.load(open('./saves/autompg_lr.pkl','rb'))



# In[ ]:





# In[ ]:





# In[ ]:




