#!/usr/bin/env python
# coding: utf-8

# ## Lista 1 - Mineração de Dados
# Pâmela Rodrigues Venturini de Souza

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('all_wc_18_players_fifa.csv', sep = ';')
df.head()


# In[2]:


#Questão 1
df.info()


# In[3]:


#Questão 2
df.drop('number', axis = 1, inplace = True)
df.info()


# In[4]:


#Questão 3
#Quantidade de instâncias
len(df)


# In[5]:


#Verificar se existe dados faltantes
df.isna().sum()


# In[6]:


#Questão 4
df.describe()


# In[7]:


#Questão 5 
sns.countplot(x = df.position);


# In[8]:


#Questão 6
#Amplitude
df.describe().loc['max'] - df.describe().loc['min']


# In[9]:


#Intervalo interquartis
iiq = df.describe().loc['75%'] - df.describe().loc['25%']
iiq


# In[10]:


#Limite mínimo
df.describe().loc['min'] - 1.5*iiq


# In[11]:


#Limite máximo
df.describe().loc['max'] + 1.5*iiq


# In[12]:


#Questão 7
#Club com maior núm. de jogadores convocados
df.club.value_counts().head(1)


# In[13]:


#Liga com maior núm. de convocados
df.league.value_counts().head(1)


# In[14]:


#Questão 8
fig, matriz = plt.subplots(nrows = 4, ncols = 2, figsize = (10,20))
sns.boxplot(data = df, y = 'height', ax = matriz[0,0], color = 'blue');
sns.histplot(data = df, y = 'height', ax = matriz[0,1], color = 'blue');

sns.boxplot(data = df, y = 'weight', ax = matriz[1,0], color = 'orange');
sns.histplot(data = df, y = 'weight', ax = matriz[1,1], color = 'orange');

sns.boxplot(data=df, y='age', ax=matriz[2,0], color='green');
sns.histplot(data=df, y='age', ax=matriz[2,1], color='green');

sns.boxplot(data=df, y='caps', ax=matriz[3,0], color='red');
sns.histplot(data=df, y='caps', ax=matriz[3,1], color='red');


# In[15]:


'''
No atributo CAPS é possível notar que há mais de 10 outliers

'''


# In[16]:


#Questão 9 
plt.figure(figsize = (7,7))
sns.scatterplot(data = df, x = 'height', y = 'weight', hue = 'position');


# In[17]:


'''
Percebe-se que há uma grande tendência de os GK estarem na parte 
superior direita, ou seja são altos e possuem um peso elevado, já os MF
possuem uma tendência a estarem na parte inferior esquerda, são mais
baixos e mais magros, já os DF e FW possuem uma variedade elevada
pois os pontos estão bem distribuidos.

'''


# In[18]:


#Questão 10
df.corr().loc['age', 'caps']


# In[19]:


'''
Sim, há uma correlação bem significativa, visto que está bem próximo de 1.

'''


# In[20]:


sns.scatterplot(data = df, x = 'age', y = 'caps');


# In[21]:


sns.lmplot(data = df, x = 'age', y = 'caps', height = 10);


# In[22]:


#Questão 11
#a) Menor média de idade
df.groupby('club').mean().age.sort_values().head()


# In[23]:


#b) Maior média de altura
df.groupby('club').mean().height.sort_values().tail(1)


# In[24]:


#c) Maior média de peso
df.groupby('club').mean().weight.sort_values().tail(1)


# In[25]:


#Questão 12
df['BMI'] = df.weight / ((df.height/100)**2)
df.head()


# In[26]:


#Questão 13
from sklearn.preprocessing import KBinsDiscretizer

#Categorizando os dados pelo método de quartis
categorizacao = KBinsDiscretizer(n_bins = 5, encode = 'ordinal', strategy = 'quantile')

categoria = categorizacao.fit_transform(df[['height', 'weight', 'age', 'caps', 'BMI']])

df_categoriazado = pd.DataFrame(categoria, columns=['height', 'weight', 'age', 'caps', 'BMI'])
df_categoriazado


# In[27]:


#Frequência da categoria: HEIGHT
df_categoriazado.height.value_counts().sort_index()


# In[28]:


#Frequência da categoria: WEIGHT
df_categoriazado.weight.value_counts().sort_index()


# In[29]:


#Frequência da categoria: AGE
df_categoriazado.age.value_counts().sort_index()


# In[30]:


#Frequência da categoria: CAPS
df_categoriazado.caps.value_counts().sort_index()

