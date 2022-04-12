#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


# In[109]:


#df = pd.read_csv("")
dfs = pd.read_excel("Data.xlsx", sheet_name=None)
dfs.keys()
plants = dfs['plants']
flight_dates =dfs['flight dates']
weather = dfs['weather']
planting = dfs['planting']


# In[110]:


#rename the weather data which column is Unnamed: 0
weather= weather.rename(columns={'Unnamed: 0':'Plant Date'})


# In[111]:


# Add flight dates from 2020 plant data.
df = plants.merge(flight_dates, how='left', on='Batch Number')
df = df.drop(['Flight Date_x'], axis=1)
#check missing data
df.info()
df


# In[112]:


#check how many 'r' is in the Remove
df['Remove'].value_counts()
#find the r
indexNames = df[ df['Remove'] == 'r' ].index
#drop the row which Remove = r
df.drop(indexNames , inplace=True)


# In[113]:


# we dont need remove anymore
#drop remove
df = df.drop(['Remove'], axis=1)


# In[114]:


#drop plant date which is null
df = df.dropna(axis='index', how='all', subset=['Plant Date'])


# In[115]:


# create days to check
df['days_to_check'] = pd.to_datetime(df['Flight Date_y'])-pd.to_datetime(df['Plant Date'])
df = df.drop(['Flight Date_y'], axis=1)


# In[116]:


#days to check column trandfer to int
a = df[:4615]['days_to_check'] / np.timedelta64(1, 'D')
type(a)


# In[117]:


#transfer seris to dataframe
a=a.to_frame()
#rename days to check to check days
df = df.rename(columns={'days_to_check':'check_days'})
#merge two data
df = pd.concat([df,a],axis = 1)
#drop the check days which did not transfer to int
df = df.drop(['check_days'], axis = 1)


# In[118]:


df.info()


# In[119]:


df.isnull().sum()


# In[120]:


#column leave fill null data in zero
df['Leaves'] = df['Leaves'].fillna(0)


# In[121]:


#fill in zero when leaves is not zero
df.loc[df.Leaves != 0 ,'Head Weight (g)'] = 0


# In[122]:


df.loc[df.Leaves != 0 ,'Radial Diameter (mm)'] = 0


# In[123]:


df.loc[df.Leaves != 0 ,'Polar Diameter (mm)'] = 0


# In[124]:


df.info()


# In[125]:


# cuculate the missing data
missing=df.isnull().sum().reset_index().rename(columns={0:'missNum'})
# cuculate the missing rate
missing['missRate']=missing['missNum']/df.shape[0]
# rank the missing rate
miss_analy=missing[missing.missRate>0].sort_values(by='missRate',ascending=False)
# miss_analy 

import matplotlib.pyplot as plt
import pylab as pl

fig = plt.figure(figsize=(18,6))
plt.bar(np.arange(miss_analy.shape[0]), list(miss_analy.missRate.values), align = 'center'
    ,color=['red','green','yellow','steelblue'])

plt.title('Histogram of missing value of variables')
plt.xlabel('variables names')
plt.ylabel('missing rate')
plt.xticks(np.arange(miss_analy.shape[0]),list(miss_analy['index']))
pl.xticks(rotation=90)
for x,y in enumerate(list(miss_analy.missRate.values)):
    plt.text(x,y+0.12,'{:.2%}'.format(y),ha='center',rotation=90)    
plt.ylim([0,1.2])
    
plt.show()


# In[126]:


df.info()


# In[127]:


#drop leak data
df= df.drop(['Diameter Ratio'], axis = 1)
df= df.drop(['Density (kg/L)'], axis = 1)


# In[128]:


#rename some of colunms
df = df.rename(columns={'Head Weight (g)':'head_weight', 'Radial Diameter (mm)':'radial_diameter','Polar Diameter (mm)':'polar_diameter','Fresh Weight (g)':'fresh_weight','Leaf Area (cm^2)':'leaf_area','days_to _check':'day_to_check'})


# In[129]:


df.head(50)


# In[130]:


weather


# In[131]:


#merge weather data
data_all = df.merge(weather, how='left', on='Plant Date')


# In[132]:


data_all.head(50)


# In[133]:


data_all[data_all["head_weight"].isnull()]
data_all = data_all.dropna(axis='index', how='all', subset=['head_weight'])


# In[134]:


# transfer datetime to float
# change them into the time passed since a specific reference time point
time0 = pd.Timestamp('2020-01-01')
data_all['Plant Date'] = (data_all['Plant Date'] - time0).values.astype(float)
data_all['Check Date'] = (data_all['Check Date'] - time0).values.astype(float)
#df['Flight Date_y'] = (df['Flight Date_y'] - time0).values.astype(float)


# In[135]:


data_all = data_all.dropna()


# In[136]:


data_all = data_all.reset_index()
data_all = data_all.drop(['index'], axis=1)


# In[137]:


data_all


# In[138]:


data_all.describe()


# In[139]:


#find the cor
#select the important feature that relate to target(polar_diameter,radial_diameter,head_weight)
import seaborn as sns
import matplotlib.pyplot as plt


# In[140]:


correlations = data_all.corr()
correlations['polar_diameter'].sort_values(ascending=False)
plt.figure(figsize=(12,10))
sns.heatmap(correlations, annot=True, cmap=plt.cm.Reds, vmax=1, vmin=-1)
plt.show()


# In[195]:


targetCorr = abs(correlations['radial_diameter'])
targetCorr = targetCorr.drop('radial_diameter')
selectedFeatures = targetCorr[targetCorr>0.15]
print(f"Number of selected features: {len(selectedFeatures)} \n\nHighly relative feature list:\n{selectedFeatures}")


# In[194]:


targetCorr = abs(correlations['polar_diameter'])
targetCorr = targetCorr.drop('polar_diameter')
selectedFeatures = targetCorr[targetCorr>0.15]
print(f"Number of selected features: {len(selectedFeatures)} \n\nHighly relative feature list:\n{selectedFeatures}")


# In[193]:


targetCorr = abs(correlations['head_weight'])
targetCorr = targetCorr.drop('head_weight')
selectedFeatures = targetCorr[targetCorr>0.15]
print(f"Number of selected features: {len(selectedFeatures)} \n\nHighly relative feature list:\n{selectedFeatures}")


# In[144]:


#low_cor = data_all[['Wind Speed [avg]','Wind Speed [max]','Leaf Wetness [time]','Square ID','Precipitation [sum]','Relative Humidity [avg]']]


# In[145]:


#drop the low correlation features
data_lowcor = data_all.drop(['Wind Speed [avg]','Wind Speed [max]','Leaf Wetness [time]','Square ID','Precipitation [sum]','Relative Humidity [avg]'], axis=1)


# In[146]:


data_lowcor.info()


# In[147]:


#check data
_ = data_lowcor.hist(bins=50, figsize=(20,15))


# In[148]:


data_lowcor.describe()


# In[149]:


#train set
x = data_all.drop(['head_weight','radial_diameter','polar_diameter'], axis=1)


# In[150]:


#test set
y = data_all[['head_weight','radial_diameter','polar_diameter']]


# In[151]:


X = data_lowcor.drop(['head_weight','radial_diameter','polar_diameter'], axis=1)


# In[152]:


Y = data_lowcor[['head_weight','radial_diameter','polar_diameter']]


# In[153]:


#prepare data for classification medel
y_class = data_all[['head_weight','radial_diameter','polar_diameter']]


# In[154]:


y_class['head_size'] =0


# In[155]:


y_class['radial_size']=0
y_class['polar_size']=0


# In[156]:



y_class.loc[y_class['head_weight']<40,'head_size'] = 0

y_class.loc[(y_class['head_weight']>=40)&(y_class['head_weight']<183),'head_size'] = 1

y_class.loc[(y_class['head_weight']>=183)&(y_class['head_weight']<375),'head_size'] = 2

y_class.loc[y_class['head_weight']>=375,'head_size'] = 3


# In[157]:


y_class.loc[y_class['radial_diameter']<75,'radial_size'] = 0

y_class.loc[(y_class['radial_diameter']>=75)&(y_class['radial_diameter']<120),'radial_size'] = 1

y_class.loc[(y_class['radial_diameter']>=120)&(y_class['radial_diameter']<140),'radial_size'] = 2

y_class.loc[y_class['radial_diameter']>=140,'radial_size'] = 3


# In[158]:


y_class.loc[y_class['polar_diameter']<80,'polar_size'] = 0

y_class.loc[(y_class['polar_diameter']>=80)&(y_class['polar_diameter']<107),'polar_size'] = 1

y_class.loc[(y_class['polar_diameter']>=107)&(y_class['polar_diameter']<130),'polar_size'] = 2

y_class.loc[y_class['polar_diameter']>=130,'polar_size'] = 3


# In[159]:


y_class = y_class.drop(['head_weight','radial_diameter','polar_diameter'], axis=1)


# In[160]:


y_class


# In[161]:


y_classh = y_class.drop(['radial_size','polar_size'], axis=1)
y_classr = y_class.drop(['head_size','polar_size'], axis=1)
y_classp = y_class.drop(['head_size','radial_size'], axis=1)


# In[162]:


from sklearn.model_selection import train_test_split, cross_val_score


# In[225]:


# Create separate training and test sets. we'll use the training set for steps 3--6
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)  


# In[226]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=50)


# In[227]:


Xc_train,Xc_test, Yc_train, Yc_test = train_test_split(X, y_class, test_size=0.3, random_state=50)


# In[228]:


xc_train,xc_test, yc_train, yc_test = train_test_split(x, y_class, test_size=0.3, random_state=50)


# In[229]:


Xh_train,Xh_test, Yh_train, Yh_test = train_test_split(X, y_classh, test_size=0.3, random_state=50)
Xr_train, Xr_test, Yr_train, Yr_test = train_test_split(X, y_classr, test_size=0.3, random_state=50)
Xp_train, Xp_test, Yp_train, Yp_test = train_test_split(X, y_classp, test_size=0.3, random_state=50)


# In[230]:


#check
print(" x_train:",len(x_train),'\n',"x_test:",len(x_test),'\n',"y_train:",len(y_train),'\n',"y_test:",len(y_test))


# In[231]:


from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


# In[232]:


ml = MultiOutputRegressor(GradientBoostingRegressor(random_state=50)).fit(x_train, y_train)


# In[233]:


ML = MultiOutputRegressor(GradientBoostingRegressor(random_state=50)).fit(X_train, Y_train)


# In[234]:


R = MultiOutputRegressor(RandomForestRegressor(random_state=50)).fit(X_train, Y_train)


# In[235]:


r = MultiOutputRegressor(RandomForestRegressor(random_state=50)).fit(x_train, y_train)


# In[236]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=50)
from sklearn.multioutput import MultiOutputClassifier


# In[237]:


multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1).fit(Xc_train, Yc_train)


# In[238]:


multi_target_foresta = MultiOutputClassifier(forest, n_jobs=-1).fit(xc_train, yc_train)


# In[239]:


predict = ml.predict(x_test)


# In[240]:


PREDICT = ML.predict(X_test)


# In[241]:


PREDICTR = R.predict(X_test)


# In[242]:


predictr = r.predict(x_test)


# In[243]:


PREDICTX = multi_target_forest.predict(Xc_test)


# In[244]:


predictx = multi_target_foresta.predict(xc_test)


# In[245]:


mean_squared_error(y_test, predict,multioutput='raw_values') 


# In[246]:


r2_score(y_test, predict, multioutput='raw_values')


# In[247]:


mean_squared_error(Y_test, PREDICT,multioutput='raw_values') 


# In[248]:


r2_score(Y_test, PREDICT, multioutput='raw_values')


# In[249]:


mean_squared_error(Y_test, PREDICTR,multioutput='raw_values') 


# In[250]:


r2_score(Y_test, PREDICTR, multioutput='raw_values')


# In[251]:


mean_squared_error(y_test, predictr,multioutput='raw_values') 


# In[252]:


r2_score(y_test, predictr, multioutput='raw_values')


# In[180]:


from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import roc_auc_score


# In[253]:


Yc_test = Yc_test.to_numpy()


# In[254]:


yc_test = yc_test.to_numpy()


# In[255]:


def example_subset_accuracy(gt, predict):
    ex_equal = np.all(np.equal(gt, predict), axis=1).astype("float32")
    return np.mean(ex_equal)


# In[256]:


example_subset_accuracy(Yc_test, PREDICTX)


# In[257]:


example_subset_accuracy(yc_test, predictx)


# In[231]:


print('MSE =', mean_squared_error(y_test, predict))
print('R^2:',ml.score(x_test, y_test))


# In[232]:


print('MSE =', mean_squared_error(Y_test, PREDICT))
print('R^2:',ML.score(X_test, Y_test))


# In[76]:


plt.scatter(y_test,predict)


# In[90]:


plt.scatter(Y_test,PREDICT)


# In[78]:


from sklearn import datasets, linear_model


# In[79]:


lr = MultiOutputRegressor(linear_model.Lasso(random_state=50)).fit(x_train, y_train)


# In[91]:


LR = MultiOutputRegressor(linear_model.Lasso(random_state=50)).fit(X_train, Y_train)


# In[92]:


predict1 = lr.predict(x_test)
PREDICT1 = LR.predict(X_test)


# In[81]:


print('MSE =', mean_squared_error(y_test, predict1))
print('R^2:',lr.score(x_test, y_test))


# In[93]:


print('MSE =', mean_squared_error(Y_test, PREDICT1))
print('R^2:',LR.score(X_test, Y_test))


# In[82]:


plt.scatter(y_test,predict1)


# In[94]:


plt.scatter(Y_test,PREDICT1)


# In[224]:


importances = r.feature_importances_
std = np.std([tree.feature_importances_ for tree in r.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
print(indices)


# In[ ]:




