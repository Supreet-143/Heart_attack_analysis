#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Dataset
# This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.
# 

# In[12]:


pip install opendatasets


# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport 
import seaborn as sns
import opendatasets as od
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


sns.set_style('darkgrid')


# In[76]:


dataset_url = 'https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset'
od.download(dataset_url)


# In[77]:


data_dir = './heart-disease-dataset'
os.listdir(data_dir)


# In[78]:


heart_diesease_df = pd.read_csv('./heart-disease-dataset/heart.csv')
heart_diesease_df


# In[79]:


heart_diesease_df['cp'].unique()


# In[80]:


heart_diesease_df.head(10)


# In[81]:


heart_diesease_df.isnull().sum()


# In[82]:


heart_diesease_df.duplicated().any()


# In[83]:


heart_diesease_df.drop_duplicates(inplace=True)
heart_diesease_df


# In[84]:


heart_diesease_df['thal'].value_counts()


# In[85]:


heart_diesease_df.drop(heart_diesease_df.index[heart_diesease_df.thal == 0], inplace=True)
heart_diesease_df['thal'].value_counts()


# In[86]:


heart_diesease_df


# In[87]:


heart_diesease_df.info()


# In[88]:


heart_diesease_df.drop(columns=['trestbps','restecg','thalach','exang', 'oldpeak', 'slope', 'ca'], inplace=True)
heart_diesease_df


# In[92]:


heart_diesease_df.describe().round(2)


# In[103]:


corrilation_matrix = heart_diesease_df.corr().round(2)


# In[104]:


sns.heatmap(corrilation_matrix, annot=True, cmap='inferno')
plt.title('Fig-1: Co-relation matrix of heart disease dataset', loc='center');


# In[110]:


chart_labels = [heart_diesease_df.target.value_counts()[1], heart_diesease_df.target.value_counts()[0]]

plt.figure(figsize=(12, 8))
plt.title('Fig-2: Pie chart showing the healthy vs at risk population')
heart_diesease_df.target.value_counts().plot(kind='pie', labels=chart_labels, autopct='%1.1f%%')
plt.legend(['At risk', 'Healthy']);


# In[111]:


heart_diesease_df


# In[116]:


heart_diesease_df.age.hist(bins=20)
plt.xlabel('Age')
plt.title('Fig-3: Age distribution among the dataset');


# In[154]:


def age_group(row):
    if row.age >= 70:
        return 70
    elif row.age >= 60:
        return 60
    elif row.age >= 50:
        return 50
    elif row.age >= 40:
        return 40
    elif row.age >= 30:
        return 30
    elif row.age >=20:
        return 20

heart_diesease_df['age_group'] = heart_diesease_df.apply(age_group, axis=1)


# In[184]:


heart_diesease_df.rename(columns={'cp':'chest_pain', 'chol':'cholestrol', 'fbs':'fast_blood_sugar', 'thal':'Thalassemia', 'age_group':'a_g'}, inplace=True)
heart_diesease_df


# In[217]:


df = heart_diesease_df.groupby(['a_g']).target.count().to_frame(name=None)
df


# In[221]:


df = df.reset_index()
print(df.columns)


# In[231]:


df.rename(columns={'a_g':'age_group', 'target':'total_patients'}, inplace=True)
df.drop(columns='index', inplace=True)


# In[232]:


df


# In[237]:


fig = plt.figure(figsize=(12,8))
df.total_patients.plot(kind='pie', labels=df.total_patients)
plt.legend(['20','30','40','50','60','70'])
plt.title('Number of patients from each age group');


# In[258]:


df1 = heart_diesease_df.groupby(['a_g', 'target'])['target'].count()
df1


# In[259]:


df1 = df1.to_frame(name=None)
print(type(df1))


# In[260]:


df1.rename(columns={'target':'targetwise_total'}, inplace=True)
df1


# In[ ]:




