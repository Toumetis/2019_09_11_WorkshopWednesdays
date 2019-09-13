#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
#  <p>

# Preparation for workshop: data preparation

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


# In[2]:


iris = load_iris()
usage_df = pd.DataFrame(iris.data, columns = iris.feature_names)
usage_df['target'] = iris.target
usage_df.shape

target names: array(['setosa', 'versicolor', 'virginica']
# In[3]:


usage_df.target.value_counts()


# In[4]:


usage_df.columns = ['12am_8am','8am_2pm','2pm_9pm','9pm_12am','target']

# swap columns to make data more realistic
usage_df = usage_df[['8am_2pm','2pm_9pm','12am_8am','9pm_12am','target']]
usage_df.columns = ['12am_8am','8am_2pm','2pm_9pm','9pm_12am','target']

# add check column
usage_df['total_kwh'] = usage_df[['12am_8am','8am_2pm','2pm_9pm','9pm_12am']].apply(lambda x: x.sum(), axis=1)


# In[5]:


# to make it more realistic, we need to get total value to around 6-7kwh per day
# half most measurements
usage_df.iloc[:,0] = round(usage_df.iloc[:,0]*(0.5),1)
#usage_df.iloc[:,1] = round(usage_df.iloc[:,1]*(0.1),1)
usage_df.iloc[:,2] = round(usage_df.iloc[:,2]*(0.5),1)

# update total
usage_df['total_kwh'] = usage_df[['12am_8am','8am_2pm','2pm_9pm','9pm_12am']].apply(lambda x: x.sum(), axis=1)

usage_df.head(2)


# In[6]:


usage_df.tail(2)


# In[7]:


# check
usage_df.total_kwh.hist();


# In[8]:


# add some null values

sample_cols = np.random.choice([0,1,2,3],size=10,replace=True)
sample_rows = np.random.choice(usage_df.index,10,replace=False)

for pos in np.arange(len(sample_cols)):
    usage_df.iloc[sample_rows[pos],sample_cols[pos]] = np.nan

sample_cols,sample_rows


# In[9]:


# check
np.where(usage_df.isnull())


# In[10]:


# shuffle data and reset index
usage_df = usage_df.sample(frac=1).reset_index(drop=True)


# In[13]:


usage_df.tail(2)


# In[14]:


path = '/Users/elenahensinger/Documents/PROJECTS/2019/09_WorkshopWednesdays_W1/datasets'

usage_df[['12am_8am','8am_2pm','2pm_9pm','9pm_12am']].to_csv(path + '/workshop_data1.csv')


# In[ ]:




