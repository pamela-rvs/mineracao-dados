#!/usr/bin/env python
# coding: utf-8

# ## Lista 3 - Mineração de Dados
# Pâmela Rodrigues Venturini de Souza

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df1 = pd.read_csv('Bejaia Region Dataset.csv', sep = ';')
df2 = pd.read_csv('Sidi-Bel Abbes Region Dataset.csv', sep = ';')


# In[2]:


df1


# In[3]:


df2


# In[4]:


df1.columns


# In[5]:


#Questão 1

df1 = df1[['Temperature', ' RH', ' Ws', 'Rain ', 'FWI']]
df2 = df2[['Temperature', ' RH', ' Ws', 'Rain ', 'FWI']]


# In[6]:


df1


# In[7]:


df2


# In[8]:


#Questão 3

fig, matriz = plt.subplots(nrows = 5, ncols = 2, figsize = (10,20))
sns.boxplot(data = df1, y = 'Temperature', ax = matriz[0,0], color='blue');
sns.histplot(data = df1, y = 'Temperature', ax = matriz[0,1], color='blue');

sns.boxplot(data = df1, y = ' RH', ax = matriz[1,0], color='orange');
sns.histplot(data = df1, y = ' RH', ax = matriz[1,1], color='orange');

sns.boxplot(data = df1, y = ' Ws', ax = matriz[2,0], color='green');
sns.histplot(data = df1, y = ' Ws', ax = matriz[2,1], color='green');

sns.boxplot(data = df1, y = 'Rain ', ax = matriz[3,0], color='red');
sns.histplot(data = df1, y = 'Rain ', ax = matriz[3,1], color='red');

sns.boxplot(data = df1, y = 'FWI', ax = matriz[4,0], color='purple');
sns.histplot(data = df1, y = 'FWI', ax = matriz[4,1], color='purple');


# In[9]:


fig, matriz = plt.subplots(nrows = 5, ncols = 2, figsize = (10,20))
sns.boxplot(data = df2, y = 'Temperature', ax = matriz[0,0], color='blue');
sns.histplot(data = df2, y = 'Temperature', ax = matriz[0,1], color='blue');

sns.boxplot(data = df2, y = ' RH', ax = matriz[1,0], color='orange');
sns.histplot(data = df2, y = ' RH', ax = matriz[1,1], color='orange');

sns.boxplot(data = df2, y = ' Ws', ax = matriz[2,0], color='green');
sns.histplot(data = df2, y = ' Ws', ax = matriz[2,1], color='green');

sns.boxplot(data = df2, y = 'Rain ', ax = matriz[3,0], color='red');
sns.histplot(data = df2, y = 'Rain ', ax = matriz[3,1], color='red');

sns.boxplot(data = df2, y = 'FWI', ax = matriz[4,0], color='purple');
sns.histplot(data = df2, y = 'FWI', ax = matriz[4,1], color='purple');


# In[10]:


df1.describe()


# In[11]:


df2.describe()


# In[12]:


#Verifica se existe dados faltantes
df1.isna().sum()


# In[13]:


df2.isna().sum()


# In[14]:


#Normalizar os dados pelo método MinMax
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df_aux = scaler.fit_transform(df1) 
df1 = pd.DataFrame(df_aux, columns = df1.columns)

df1


# In[15]:


#Mínimo
df1.min()


# In[16]:


#Máximo
df1.max()


# In[17]:


#Informações sobre correlação 
df1.iloc[:,0:14].corr()


# In[18]:


y1 = df1.FWI
y2 = df2.FWI


# In[19]:


#Questão 4

#Árvore de Decisão - Regressão

from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor(random_state = 5)

DT.fit(df1, y1)


# In[20]:


DT_pred = DT.predict(df2)


# In[21]:


len(DT_pred), len(y2)


# In[22]:


#Importando métricas de avaliação de modelos de regressão
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# In[23]:


print('Métricas de regressão')
print(f'MAE = {mean_absolute_error(y2, DT_pred):.3f}')
print(f'MSE = {mean_squared_error(y2, DT_pred):.3f}')
print(f'RMSE = {math.sqrt(mean_squared_error(y2, DT_pred)):.3f}')
print(f'MAPE = {mean_absolute_percentage_error(y2, DT_pred):.3f}')


# In[24]:


#Floresta aleatória - Random forest
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators = 500)


# In[25]:


RF.fit(df1, y1)


# In[26]:


RF_pred = RF.predict(df2)


# In[27]:


print('Métricas de regressão')
print(f'MAE = {mean_absolute_error(y2, RF_pred):.3f}')
print(f'MSE = {mean_squared_error(y2, RF_pred):.3f}')
print(f'RMSE = {math.sqrt(mean_squared_error(y2, RF_pred)):.3f}')
print(f'MAPE = {mean_absolute_percentage_error(y2, RF_pred):.3f}')


# In[28]:


#Redes Neurais Artificiais - MLP
from sklearn.neural_network import MLPRegressor

MLP = MLPRegressor(max_iter = 500)


# In[29]:


MLP.fit(df1, y1)


# In[30]:


MLP_pred = MLP.predict(df2)


# In[31]:


print('Métricas de regressão')
print(f'MAE = {mean_absolute_error(y2, MLP_pred):.3f}')
print(f'MSE = {mean_squared_error(y2, MLP_pred):.3f}')
print(f'RMSE = {math.sqrt(mean_squared_error(y2, MLP_pred)):.3f}')
print(f'MAPE = {mean_absolute_percentage_error(y2, MLP_pred):.3f}')


# In[32]:


#Questão 5
erro = []
#a) Floresta Aleatoria - Utilizando 100
RF = RandomForestRegressor(n_estimators = 100)
RF.fit(df1, y1)
RF_pred = RF.predict(df2)
erro.append(mean_squared_error(y2, RF_pred))
print(f'MSE = {mean_squared_error(y2, RF_pred):.3f}')


# In[33]:


#Floresta Aleatoria - Utilizando 200
RF = RandomForestRegressor(n_estimators = 200)
RF.fit(df1, y1)
RF_pred = RF.predict(df2)
erro.append(mean_squared_error(y2, RF_pred))
print(f'MSE = {mean_squared_error(y2, RF_pred):.3f}')


# In[34]:


#Floresta Aleatoria - Utilizando 300
RF = RandomForestRegressor(n_estimators = 300)
RF.fit(df1, y1)
RF_pred = RF.predict(df2)
erro.append(mean_squared_error(y2, RF_pred))
print(f'MSE = {mean_squared_error(y2, RF_pred):.3f}')


# In[35]:


#Floresta Aleatoria - Utilizando 500
RF = RandomForestRegressor(n_estimators = 500)
RF.fit(df1, y1)
RF_pred = RF.predict(df2)
erro.append(mean_squared_error(y2, RF_pred))
print(f'MSE = {mean_squared_error(y2, RF_pred):.3f}')


# In[36]:


#Floresta Aleatoria - Utilizando 1000
RF = RandomForestRegressor(n_estimators = 1000)
RF.fit(df1, y1)
RF_pred = RF.predict(df2)
erro.append(mean_squared_error(y2, RF_pred))
print(f'MSE = {mean_squared_error(y2, RF_pred):.3f}')


# In[37]:


sns.lineplot(x = [100, 200, 300, 500, 1000], y = erro);


# In[38]:


menor = min(erro)
menor


# In[39]:


erro = []
#b) Redes neurais - Utilizando 200
MLP = MLPRegressor(max_iter = 200)
MLP.fit(df1, y1)
MLP_pred = MLP.predict(df2)
erro.append(mean_squared_error(y2, MLP_pred))
print(f'MSE = {mean_squared_error(y2, MLP_pred):.3f}')


# In[40]:


#Redes neurais - Utilizando 300
MLP = MLPRegressor(max_iter = 300)
MLP.fit(df1, y1)
MLP_pred = MLP.predict(df2)
erro.append(mean_squared_error(y2, MLP_pred))
print(f'MSE = {mean_squared_error(y2, MLP_pred):.3f}')


# In[41]:


#Redes neurais - Utilizando 500
MLP = MLPRegressor(max_iter = 500)
MLP.fit(df1, y1)
MLP_pred = MLP.predict(df2)
erro.append(mean_squared_error(y2, MLP_pred))
print(f'MSE = {mean_squared_error(y2, MLP_pred):.3f}')


# In[42]:


#Redes neurais - Utilizando 1000
MLP = MLPRegressor(max_iter = 1000)
MLP.fit(df1, y1)
MLP_pred = MLP.predict(df2)
erro.append(mean_squared_error(y2, MLP_pred))
print(f'MSE = {mean_squared_error(y2, MLP_pred):.3f}')


# In[43]:


#Redes neurais - Utilizando 2000
MLP = MLPRegressor(max_iter = 2000)
MLP.fit(df1, y1)
MLP_pred = MLP.predict(df2)
erro.append(mean_squared_error(y2, MLP_pred))
print(f'MSE = {mean_squared_error(y2, MLP_pred):.3f}')


# In[44]:


sns.lineplot(x = [200, 300, 500, 1000, 2000], y = erro);


# In[45]:


menor = min(erro)
menor


# In[46]:


#c) A técnica que apresentou o melhor resultado foi a de Redes Neurais, 
# pois apresentou o menor erro quadrático médio em sua iterações


# In[ ]:




