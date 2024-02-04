#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORTINNG LIBRARIES

import numpy as np
import pandas as pd

#2 variables
True_news=pd.read_csv("truenews.csv")
Fake_news=pd.read_csv('fakenews.csv')


# In[2]:


True_news #retrieving data 


# In[3]:


True_news['label']=0   #to add a column to both type of news
Fake_news['label']=1
True_news


# In[4]:


Fake_news


# In[5]:


dataset1 =True_news[['name','label']]
dataset2 =Fake_news[['name','label']]
dataset=pd.concat([dataset1,dataset2])


# In[6]:


dataset2


# In[7]:


dataset=pd.concat([dataset1,dataset2])


# In[8]:


dataset


# In[9]:


dataset.shape  #81 datapoints and 2 is number of columns


# In[10]:


#to check nullvalue in dataset before preprocessing
#missing values
dataset.isnull().sum()
#NO VALUES ARE NULL -output


# In[11]:


dataset["label"].value_counts()   #0 has 41 datapoints
                                  #1 has 40 datapoints


# In[12]:


#shuffle 0 & 1 to break biasing
#RESAMPLING
dataset=dataset.sample(frac =1)


# In[13]:


dataset


# In[14]:


#natural learning process(machine understands only 0's and 1's) -- nltk

import nltk
import re #datacleaning library
from nltk.corpus import stopwords       
from nltk.stem import WordNetLemmatizer 


# In[15]:


import nltk #installation of nltk
nltk.download()


# In[16]:


nltk.download('wordnet')


# In[17]:


nltk.download("stopwords")


# In[18]:


nltk.download("stopwords")


# In[19]:


import nltk
import re #datacleaning library
from nltk.corpus import stopwords       
from nltk.stem import WordNetLemmatizer 
ps=WordNetLemmatizer()
stopwords=stopwords.words("english")
nltk.download('wordnet')


# In[20]:


#CLEANING PROCESS
def clean_row(row):
    row=row.lower() #all textual charaters to lower
    row=re.sub('[^a-zA-Z]',' ',row) #spl characters & numbers removing
    
    token=row.split() #by lemmatizing we have to give token
    
    news=[ps.lemmatize(word) for word in token if not word in stopwords] #lemmatize
    
    cleanned_news=' '.join(news)  #join all tokens since tokens are given for each lem.words
    
    return cleanned_news



# ###lemmatizing--> a,an,the,and ---stop words ,so we have to lemmatize and remove all stop words
# #1.my name is nithika and my field is AI
# 
# #2.my name is nithika field AI
#    0   1    2  3      4     5
# #PROCESS
# ##my , is ---> stopwords removed
# 
# 
# ##firstly whole sentence into tokens and it removes repetetive tokens eg.my,is
# ##machine learning model takes the token vector not the textual word''''''

# In[21]:


dataset['name']


# In[23]:


dataset['name'] = dataset["name"].apply(lambda x :clean_row(x))


# In[24]:


dataset['name'] #clean data


# In[ ]:


#text into vectors , vectors into ML


# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer #tis lib also converts data into lower case but already done keep it 0
vectorizer=TfidfVectorizer(max_features=500,lowercase=False,ngram_range=(1,2))


# In[27]:


#splitting for training and testing  

x=dataset.iloc[:42,0]
y=dataset.iloc[:42,1]


# In[28]:


x


# In[29]:


y


# In[36]:


from sklearn.model_selection import train_test_split
train_data,test_data,train_label,test_label = train_test_split(x,y,test_size=2,random_state= 0)
vec_train_data = vectorizer.fit_transform(train_data)


# In[37]:


len(y)


# In[ ]:


#vectorizing train data only


# In[42]:


vec_train_data=vectorizer.fit_transform(train_data)
vec_train_data=vec_train_data.toarray()


# In[44]:


type(vec_train_data),vec_test_data.shape


# In[45]:


vec_test_data=vectorizer.fit_transform(test_data)
vec_test_data=vec_test_data.toarray()


# In[47]:


vec_train_data.shape


# In[ ]:




