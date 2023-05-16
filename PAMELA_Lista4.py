#!/usr/bin/env python
# coding: utf-8

# ## Lista 4 - Mineração de Dados
# Pâmela Rodrigues Venturini de Souza

# In[25]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('ruspini.csv', sep = ',')


# In[26]:


def figura():
    tamanho = plt.figure(figsize = [12,10])
    estilo = sns.set(font_scale = 1.3, style = 'white')
    return tamanho, estilo


# In[27]:


df


# In[28]:


df.columns


# In[29]:


df.describe()


# In[30]:


#Questão 1

#Verifica se existe dados faltantes
df.isna().sum()


# In[31]:


figura()
sns.heatmap(df.isna())


# In[8]:


#Questão 2

figura()
sns.pairplot(df);


# In[9]:


sns.pairplot(df);


# #Aparentemente este conjunto podem ser separados em 4 clusters

# In[10]:


x = df.drop("Unnamed: 0", axis=1)
y = df['Unnamed: 0']


# In[11]:


x


# In[12]:


y


# In[13]:


#Questão 3 e 4 - 

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


# In[14]:


# Questão 5 
#Técnica k-means
silhueta_kmeans = list()

for i in [2,3,4,5,6]:
    kmeans = KMeans(n_clusters = i)
    print("*** KMeans ", i, "***")
    print("FIT: ", kmeans.fit(x))
    print("INERTIA: ", kmeans.inertia_)
    print("CLUSTER_CENTERS: ", kmeans.cluster_centers_)
    print("LABELS: ", kmeans.labels_)
    print("SILHUETA: ", silhouette_score(x, kmeans.labels_), "\n\n")
    silhueta_kmeans.append(silhouette_score(x, kmeans.labels_))


# In[15]:


sns.set_style('whitegrid')

figura()
plt.plot([2,3,4,5,6],silhueta_kmeans, marker='o');


# In[16]:


# Agrupamento hierárquico - Método aglomerativo
silhueta_hierarquico = list()

for i in [2,3,4,5,6]:
    hierarquico = AgglomerativeClustering(n_clusters = i, linkage = "ward")
    print("*** Agrupamento Hierárquico ", i, "***")
    print("FIT: ", hierarquico.fit(x))
    print("LABELS: ", hierarquico.labels_)
    print("SILHUETA: ", silhouette_score(x, hierarquico.labels_), "\n\n")
    silhueta_hierarquico.append(silhouette_score(x, hierarquico.labels_))


# In[17]:


sns.set_style('whitegrid')

figura()
plt.plot([2,3,4,5,6],silhueta_hierarquico, marker='o');


# In[18]:


sns.set_style('whitegrid')

figura()
plt.plot([2,3,4,5,6],silhueta_kmeans, marker = 'o', color = 'blue');
plt.plot([2,3,4,5,6],silhueta_hierarquico, marker = 'o', color = 'purple');


# In[19]:


#Em ambos os métodos a melhor opção será ter 4 clusters


# In[20]:


#Questão 6 

kmeans = KMeans(n_clusters = 4)
kmeans.fit(x)
kmeans.inertia_
kmeans.cluster_centers_
kmeans.labels_

sns.scatterplot(x = x.x, y = x.y, hue = kmeans.labels_, palette = 'Set1');


# In[21]:


hierarquico = AgglomerativeClustering(n_clusters = 4, linkage = "ward")
hierarquico.fit(x)
hierarquico.labels_

sns.scatterplot(x = x.x, y = x.y, hue = hierarquico.labels_, palette = 'Set1');


# In[22]:


#Questão 7

#Técnica k-means
inercia_kmeans = list()

for i in [2,3,4,5,6]:
    kmeans = KMeans(n_clusters = i)
    print("*** KMeans ", i, "***")
    print("FIT: ", kmeans.fit(x))
    print("INERTIA: ", kmeans.inertia_, "\n\n")
    inercia_kmeans.append(kmeans.inertia_)


# In[23]:


figura()
plt.plot([2,3,4,5,6],inercia_kmeans, marker = 'o');


# In[24]:


#É possível notar que pela métrica da inércia e utilizando do método do cotovelo 
#a quantidade ideal de clusters é 4, assim como sugerido na questão 5.


# In[ ]:





# In[ ]:




