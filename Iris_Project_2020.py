#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


print(np.__version__)
print(pd.__version__)
print(sns.__version__)
import sys
print(sys.version)


# In[9]:


df = pd.read_csv('Desktop/ML/Data/iris.data', header=None)
col_name = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
df.columns = col_name
df.head()


# In[10]:


iris = sns.load_dataset('iris')
iris.head()


# In[13]:


df.describe()


# In[14]:


iris.describe()


# In[15]:


print(df.info())
print(iris.info())


# In[18]:


print(df.groupby('class').size())
print(iris.groupby('species').size())


# In[20]:


sns.pairplot(iris, hue='species', height=3, aspect=1)


# In[21]:


iris.hist(edgecolor='black', linewidth=1.2, figsize=(12,8))
plt.show()


# In[22]:


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.violinplot(x='species', y='sepal_length', data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species', y='sepal_width', data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species', y='petal_length', data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species', y='petal_width', data=iris)


# In[23]:


iris.boxplot(by='species', figsize=(12,8))


# In[24]:


pd.plotting.scatter_matrix(iris, figsize=(12,10))
plt.show()


# In[25]:


sns.pairplot(iris, hue="species", diag_kind="kde")


# In[ ]:




