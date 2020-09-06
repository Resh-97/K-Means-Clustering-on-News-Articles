#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries here
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  
from sklearn.decomposition import PCA


# In[2]:


Data = pd.read_json("NewsArticles.json")
NewsArticle_df = pd.DataFrame(Data,columns=Data.columns)
NewsArticle_df.reset_index
NewsArticle_df.head()


# In[3]:


#Extracting the vectors into columns
vectors = NewsArticle_df.Vector.apply(pd.Series)

#kmeans
kmeans = KMeans(n_clusters=5,random_state=10)
kmeans.fit(vectors)
labels_df =pd.DataFrame(kmeans.labels_ , columns = ['labels_pre_pca'])
cluster_result = pd.concat([vectors,labels_df],axis =1)
NewsArticle_result = pd.concat([NewsArticle_df,labels_df],axis =1)

SSE_pre_pca = (kmeans.inertia_).round(2)
print(SSE_pre_pca) #Task 1 


# In[4]:


pca = PCA(n_components=100)
pca.fit(vectors)
vectors_2d =pca.transform(vectors)
vectors_post_pca = pd.DataFrame(vectors_2d)
kmeans_afterPCA=KMeans(n_clusters=5,random_state=10)
kmeans_afterPCA.fit(vectors_post_pca)

labels_df_post_pca =pd.DataFrame(kmeans_afterPCA.labels_ , columns = ['labels_post_pca'])
cluster_result = pd.concat([cluster_result,labels_df_post_pca],axis =1)
NewsArticle_result = pd.concat([NewsArticle_result,labels_df_post_pca],axis =1)

SSE_post_pca = (kmeans_afterPCA.inertia_).round(2)
print(SSE_post_pca) #Task 2


# In[8]:


size_pre = cluster_result.groupby("labels_pre_pca").count()[0]
size_post = cluster_result.round(2).groupby("labels_post_pca").count()[0]
for k in range(0,5):
    if size_pre[k] == max(size_pre):
        high_count_pre_pca = k
        count_pre = size_pre[k]
print('Cluster having the highest value of count (before PCA) is {}'.format(high_count_pre_pca),"with count {}".format(count_pre)) #Task 3


# In[10]:


for k in range(0,5):
    if size_post[k] == max(size_post):
        high_count_post_pca = k
        count_post =size_post[k]
print('Cluster having the highest value of count (after PCA) is {}'.format(high_count_post_pca),"with count {}".format(count_post)) #Task 4


# In[11]:


entertainment_pre = NewsArticle_result[(NewsArticle_result['labels_pre_pca']==3)]
word_list=entertainment_pre['Preprocessed-Article'][0].rsplit(" ")
for x in word_list:
    if (x == ' ' ) or (x == '' ):
        word_list.remove(x)
print(word_list[49]) #Task 5


# In[12]:


entertainment_post = NewsArticle_result[(NewsArticle_result['labels_post_pca']==3)]
entertainment_post.reset_index(inplace=True)
#print(entertainment_post)
word_list_post=entertainment_post['Preprocessed-Article'][0].rsplit(" ")
for x in word_list_post:
    if (x == ' ' ) or (x == '' ):
        word_list_post.remove(x)
print(word_list_post[49])     #Task 6


# In[ ]:


#Saving the outputs

result = [SSE_pre_pca,SSE_post_pca,high_count_pre_pca,count_pre,high_count_post_pca,count_post,word_list[49],word_list_post[49]]
result=pd.DataFrame(result)
#writing output to output.csv
result.to_csv('/output.csv', header=False, index=False)

