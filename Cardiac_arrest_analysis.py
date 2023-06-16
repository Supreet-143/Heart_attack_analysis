#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Dataset
# This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.
# 

# In[1]:


pip install opendatasets


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport 
import seaborn as sns
import opendatasets as od
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set_style('darkgrid')


# In[4]:


dataset_url = 'https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset'
od.download(dataset_url)


# In[5]:


data_dir = './heart-disease-dataset'
os.listdir(data_dir)


# In[6]:


heart_diesease_df = pd.read_csv('./heart-disease-dataset/heart.csv')
heart_diesease_df


# In[7]:


heart_diesease_df['cp'].unique()


# In[8]:


heart_diesease_df.head(10)


# In[9]:


heart_diesease_df.isnull().sum()


# In[10]:


heart_diesease_df.duplicated().any()


# In[11]:


heart_diesease_df.drop_duplicates(inplace=True)
heart_diesease_df


# In[12]:


heart_diesease_df['thal'].value_counts()


# In[13]:


heart_diesease_df.drop(heart_diesease_df.index[heart_diesease_df.thal == 0], inplace=True)
heart_diesease_df['thal'].value_counts()


# In[14]:


heart_diesease_df


# In[15]:


heart_diesease_df.info()


# In[16]:


heart_diesease_df.drop(columns=['trestbps','restecg','thalach','exang', 'oldpeak', 'slope', 'ca'], inplace=True)
heart_diesease_df


# In[17]:


heart_diesease_df.describe().round(2)


# In[18]:


corrilation_matrix = heart_diesease_df.corr().round(2)


# In[19]:


sns.heatmap(corrilation_matrix, annot=True, cmap='inferno')
plt.title('Fig-1: Co-relation matrix of heart disease dataset', loc='center');


# In[20]:


chart_labels = [heart_diesease_df.target.value_counts()[1], heart_diesease_df.target.value_counts()[0]]

plt.figure(figsize=(12, 8))
plt.title('Fig-2: Pie chart showing the healthy vs at risk population')
heart_diesease_df.target.value_counts().plot(kind='pie', labels=chart_labels, autopct='%1.1f%%')
plt.legend(['At risk', 'Healthy']);


# In[21]:


heart_diesease_df


# In[22]:


heart_diesease_df.age.hist(bins=20)
plt.xlabel('Age')
plt.title('Fig-3: Age distribution among the dataset');


# In[23]:


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


# In[24]:


heart_diesease_df.rename(columns={'cp':'chest_pain', 'chol':'cholestrol', 'fbs':'fast_blood_sugar', 'thal':'Thalassemia', 'age_group':'a_g'}, inplace=True)
heart_diesease_df


# In[25]:


df = heart_diesease_df.groupby(['a_g']).target.count().to_frame(name=None)
df


# In[26]:


df = df.reset_index()
print(df.columns)


# In[27]:


df.rename(columns={'a_g':'age_group', 'target':'total_patients'}, inplace=True)



# In[28]:


df


# In[29]:


fig = plt.figure(figsize=(12,8))
df.total_patients.plot(kind='pie', labels=df.total_patients)
plt.legend(['20','30','40','50','60','70'])
plt.title('Number of patients from each age group');


# In[30]:


df1 = heart_diesease_df.groupby(['a_g', 'target'])['target'].count()
df1


# In[31]:


df1 = df1.to_frame(name=None)
print(type(df1))


# In[32]:


df1.rename(columns={'target':'targetwise_total'}, inplace=True)
df1


# In[33]:


heart_diesease_df


# In[34]:


Total_female_population = heart_diesease_df.query('sex==0')['sex'].count()


# In[35]:


Total_vulnarable_female = heart_diesease_df.query('sex==0 and target==1')['sex'].count()


# In[36]:


perct_vul_female = (Total_vulnarable_female / Total_female_population) * 100


# In[37]:


perct_vul_female


# In[38]:


Total_male_population = heart_diesease_df.query('sex==1')['sex'].count()
Total_vulnarable_male = heart_diesease_df.query('sex==1 and target==1')['sex'].count()
perct_vul_male = (Total_vulnarable_male / Total_male_population) * 100
perct_vul_male


# In[39]:


heart_diesease_df


# In[40]:


High_risk_df = heart_diesease_df.query('target==1')
Low_risk_df = heart_diesease_df.query('target==0')


# In[41]:


High_risk_df.head(10)


# In[42]:


g = sns.FacetGrid(High_risk_df, col='sex', margin_titles=False, height=6)
g.map(sns.histplot, 'age', color='red')
g.add_legend()
g.fig.suptitle('Fig-4: Age Distribution in higher risk patients');


# In[43]:


High_risk_df['sex'] = High_risk_df['sex'].replace([0, 1], ['Female', 'Male'])
High_risk_df.rename(columns={'target':'high_risk'}, inplace=True)
High_risk_df


# In[44]:


df2 = High_risk_df.groupby('sex').high_risk.count().to_frame(name=None)
df2.reset_index(inplace=True)
df2


# In[52]:


df3 = High_risk_df.groupby(['sex', 'a_g']).high_risk.count().to_frame(name= None)
df3


# In[58]:


sns.countplot(x='sex', hue='target', data=heart_diesease_df)
plt.xticks([1, 0], ['Male', 'Female'])
plt.legend(labels=['Less Risky', 'More Risky'])
plt.title('Fig: 5 Comparing cardiac arrest risks among genders', loc='center');


# In[78]:


chest_pain_types = heart_diesease_df.groupby('chest_pain')['chest_pain'].count()
chest_pain_types


# In[85]:


heart_diesease_df


# In[91]:


heart_diesease_df['chest_pain'].unique()


# In[94]:


sns.countplot(x=heart_diesease_df['chest_pain'])
plt.xticks([0,1,2,3], ['Typical angina', 'Atypical angina', 'Non-angina', 'Asymptotic'])
plt.title('Types of chest pain')
plt.xlabel('Chest_pain_type');


# In[ ]:





# In[ ]:




