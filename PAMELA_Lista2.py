#!/usr/bin/env python
# coding: utf-8

# ## Lista 2 - Mineração de Dados
# Pâmela Rodrigues Venturini de Souza

# In[1]:


#Questão 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('churn.csv')
df.head(10)


# In[2]:


df.info()


# In[3]:


df.isnull().sum()


# In[4]:


#Questão 2
df.drop('account_length', axis = 1, inplace = True)
df.drop('state', axis = 1, inplace = True)
df.columns


# In[5]:


plt.figure(figsize=(12,10));
sns.heatmap(df.isna());


# In[6]:


#a. Faça a binarização dos atributos nominais
bin_df = pd.get_dummies(df[['international_plan', 'voice_mail_plan', 'churn']], drop_first=True)
bin_df


# In[7]:


bin_dfP = pd.get_dummies(df['area_code'])
bin_dfP


# In[8]:


df_bin = pd.concat([df.drop(['area_code','international_plan', 'voice_mail_plan', 'churn'],axis = 1),bin_df, bin_dfP], axis = 1)
df_bin


# In[9]:


#b. Normalize os dados pelo método MinMax
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_bin_noraux = scaler.fit_transform(df_bin) 
df_bin_nor = pd.DataFrame(df_bin_noraux, columns = df_bin.columns)

df_bin_nor


# In[10]:


#Mínimo
df_bin_nor.min()


# In[11]:


#Máximo
df_bin_nor.max()


# In[12]:


#c. Faça uma exploração inicial com informações sobre correlação (dados numéricos apenas), análise do balanceamento das classes, tabelas de contingência, entre outras análises.
df_bin_nor.iloc[:,0:14].corr()


# In[13]:


plt.figure(figsize = (12,10));
sns.heatmap(df_bin_nor.iloc[:,0:14].corr());


# In[14]:


df_bin_nor['churn_yes'].value_counts()


# In[15]:


#d. Prepare o modelo para a validação holdout, considerando 70% do conjunto para treinamento e parâmetro random_state = 42
from sklearn.model_selection import train_test_split

x = df_bin_nor.drop('churn_yes', axis = 1)
y = df_bin_nor.churn_yes

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.7, random_state = 42)


# In[16]:


#e. Use os métodos de árvores de decisão e k-vizinhos mais próximos para classificar o conjunto
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

#Árvore de Decisão
DT = DecisionTreeClassifier(max_depth = 5)
DT.fit(x_train, y_train)


# In[17]:


DT_pred = DT.predict(x_test)
DT_pred


# In[18]:


print(classification_report(y_test, DT_pred))


# In[19]:


print(confusion_matrix(y_test, DT_pred))


# In[20]:


#K-vizinho mais próximo
KNN = KNeighborsClassifier() 

KNN.fit(x_train, y_train)
KNN_pred = KNN.predict(x_test)

KNN_pred


# In[21]:


print(classification_report(y_test, KNN_pred))


# In[22]:


print(confusion_matrix(y_test, KNN_pred))


# #f. A partir dos resultados obtidos pelos modelos, também responda
# 
# #i. Qual obteve a maior acurácia?
# * A Árvore de Decisão obteve uma acurácia de 0.93, sendo ela maior que a do K-vizinhos mais próximo que obteve 0.88 de acurácia.

# #ii. Qual modelo obteve a maior sensibilidade (recall) e precisão?
# * A Árvore de Decisão obteve Recall de 0.81 e Precisão de 0.89, maior que a do K-vizinhos mais próximo obteve Recall de 0.61 e Precisão 0.87.
# 

# In[23]:


# iii. Qual retornou a melhor média harmônica entre recall e precisão?

from scipy import stats

#Média Harmónica - Árvore de Decisão
stats.hmean([0.89,0.81])


# In[24]:


#Média Harmónica - K-vizinho mais próximo
stats.hmean([0.87,0.61])


# * A Árvore de Decisão obteve maior Média harmônica, sendo de 0.8481

# In[25]:


# iv. Mostre quais foram as frequências de Verdadeiros Positivos (VP), Verdadeiros Negativos (VN), Falsos Positivos (FP) e Falsos Negativos (FN)

#Árvore de Decisão
print(confusion_matrix(y_test, DT_pred))


# In[26]:


#K-vizinho mais próximo
print(confusion_matrix(y_test, KNN_pred))

