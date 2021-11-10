#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


dia = pd.read_csv(r'C:\Users\Vignesh Chowdary\OneDrive\Documents\Downloads\Diabetes.csv')


# In[3]:


dia.head()


# In[4]:


dia.describe()


# In[5]:


dia.info()


# In[6]:


dia.isnull().sum()


# In[7]:


sns.distplot(dia.Glucose , kde= False ,)


# In[10]:


plt.figure(figsize=(8,5))
sns.distplot(dia[dia['Outcome']==0].Glucose,kde=False)
sns.distplot(dia[dia['Outcome']==1].Glucose,kde=False)
plt.show()


# In[11]:


sns.boxplot(x = 'Outcome' , y='Glucose', data= dia)
plt.show()


# In[12]:


dia[dia['Outcome']==1].describe()


# In[13]:


sns.lmplot(x = 'Age' , y='Glucose',data =dia)
plt.show()


# In[14]:


sns.lmplot(x = 'Age' , y='Glucose',data =dia , hue= 'Outcome')
plt.show()


# In[15]:


cormat= dia.corr()


# In[16]:


cormat


# In[17]:


sns.heatmap(cormat, annot=True)
plt.show()


# In[18]:


df1 = dia[['Glucose','BMI','DiabetesPedigreeFunction','Outcome']]


# In[19]:


df1


# In[21]:


x = dia.iloc[:,0:7]
x.head()


# In[22]:


y = dia.iloc[:,7:]
y.head()


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train ,x_test , y_train , y_test = train_test_split(x , y , test_size = 0.4,random_state=7 )


# In[25]:


x_train.head()


# In[26]:


dia.shape


# In[27]:


x_train.shape


# In[28]:


x_test.shape


# In[29]:


x_test.head()


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


model= LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)


# In[32]:


pred


# In[33]:


result = x_test


# In[34]:


result.head()


# In[35]:


result['Actual']= y_test
result['Predicted']= pred


# In[36]:


result.head(20)


# In[37]:


doctor = model.predict([[128,78,0,0,21.1,0.268,55]])
if doctor==0:
 print('Congratulation ! you are not diabetic')
else :
 print('you are diabetic')


# In[38]:


doctor = model.predict([[160,78,0,0,21.1,0.268,55]]) # 160-161
if doctor==0:
 print('Congratulation ! you are not diabetic')
else :
 print('you are diabetic')


# In[ ]:





# In[ ]:




