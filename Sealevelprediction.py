# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:46:01 2018

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score, mean_squared_error,mean_squared_log_error   
 
data = pd.read_csv('E:\Mypythoncodes\Global_Warming_Indicator.csv')
'''
sealevel = data['Adjusted sea level']
features= data.drop('Adjusted sea level ', axis = 1)
'''
features = data.iloc[:,0:3]
feature = np.array(features)
sealevels = data.iloc[:,[3]]
sealevel = np.array(sealevels)

print(' dataset has {} data points with {} variables each.'.format(*data.shape))


minimum_sealevel = np.min(sealevel)
maximum_sealevel = np.max(sealevel)
mean_sealevel = np.mean(sealevel)
median_sealevel = np.median(sealevel)
std_sealevel = np.std(sealevel)
print("Statistics for Carbon emition,Temperature, Adjusted sea level  dataset:")
print("Minimum Adjusted sea level : {:,.2f} inchs".format(minimum_sealevel))
print("Maximum Adjusted sea level : {:,.2f} inchs".format(maximum_sealevel))
print("Mean Adjusted sea level : {:,.2f} inchs".format(mean_sealevel))
print("Median Adjusted sea level  {:,.2f} inchs".format(median_sealevel))
print("Standard deviation of Adjusted sea level : {:,.2f} inchs".format(std_sealevel))


sum_exs1=sum_ms1=sum_r21=sum_msle1=0; 
sum_exs2=sum_ms2=sum_r22=sum_msle2=0; 
sum_exs3=sum_ms3=sum_r23=sum_msle3=0;
sum_exs4=sum_ms4=sum_r24=sum_msle4=0; 
sum_exs5=sum_ms5=sum_r25=sum_msle5=0;  

def performance_measure(y_test,y_pred):
      
      explained_variance = explained_variance_score(y_test, y_pred)
      mean_squared = mean_squared_error(y_test, y_pred)
      mean_squared_log = mean_squared_log_error(y_test, y_pred)
      r_squared = r2_score(y_test, y_pred)
      print("Performance Metric...")
      print('explained variance = {}'.format(explained_variance))
      print('mse = {}'.format(mean_squared))
      print('r2 = {}'.format(r_squared))
      print('msle = {}'.format(mean_squared_log))
      
      return explained_variance,mean_squared,mean_squared_log,r_squared
     
'''
def sum_of_performance_metric(explained_variance,mean_squared,mean_squared_log,r_squared):
     
       sum_exs=sum_ms=sum_r2=sum_msle=0; 
       sum_exs = sum_exs + mean_squared
       sum_ms = sum_ms + explained_variance
       sum_r2 = sum_r2 + mean_squared_log
       sum_msle = sum_msle + r_squared
       
       return sum_exs,sum_ms,sum_r2,sum_msle
'''       

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler().fit(feature)
feature= scaler.transform(feature)           
  
X = np.array(features)
y = np.array(sealevel)  
kf=KFold(n_splits=5, random_state=None, shuffle=True)

for train_index, test_index in kf.split(X,y):
  
  #print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  
  
  regressor = RandomForestRegressor(n_estimators=500)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  explained_variance1,mean_squared1,mean_squared_log1,r_squared1 = performance_measure(y_test,y_pred)
  #sum_of_performance_metric(explained_variance1,mean_squared1,mean_squared_log1,r_squared1)
  sum_exs1 = sum_exs1 + explained_variance1 
  sum_ms1 = sum_ms1 + mean_squared1
  sum_r21 = sum_r21 + r_squared1 
  sum_msle1 = sum_msle1 + mean_squared_log1
  
  regressor = ExtraTreesRegressor(n_estimators=500)
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  explained_variance2,mean_squared2,mean_squared_log2,r_squared2 = performance_measure(y_test,y_pred)

  sum_exs2 = sum_exs2 + explained_variance2
  sum_ms2 = sum_ms2 + mean_squared2
  sum_r22 = sum_r22 + r_squared2
  sum_msle2 = sum_msle2 + mean_squared_log2

    
  regressor = DecisionTreeRegressor()
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  explained_variance3,mean_squared3,mean_squared_log3,r_squared3 = performance_measure(y_test,y_pred)

  sum_exs3 = sum_exs3 + explained_variance3
  sum_ms3 = sum_ms3 + mean_squared3
  sum_r23 = sum_r23 + r_squared3
  sum_msle3 = sum_msle3 + mean_squared_log3
  
  regressor = svm.SVR()
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  explained_variance4,mean_squared4,mean_squared_log4,r_squared4 = performance_measure(y_test,y_pred)

  sum_exs4 = sum_exs4 + explained_variance4
  sum_ms4 = sum_ms4 + mean_squared4
  sum_r24 = sum_r24 + r_squared4
  sum_msle4 = sum_msle4 + mean_squared_log4
  
  regressor =  AdaBoostRegressor()
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  explained_variance5,mean_squared5,mean_squared_log5,r_squared5 = performance_measure(y_test,y_pred)

  sum_exs5 = sum_exs5 + explained_variance5
  sum_ms5 = sum_ms5 + mean_squared5
  sum_r25 = sum_r25 + r_squared5
  sum_msle5 = sum_msle5 + mean_squared_log5
  
print(" ..........Average of 5 fold performance............ ") 
print("Model 1: RandomForestRegressor") 
avg_RandomForest_exs = (sum_exs5)/5
avg_RandomForest_ms = (sum_ms5)/5
avg_RandomForest_r2 = (sum_r25)/5
avg_RandomForest_msle = (sum_msle5)/5

print('explained variance = {}'.format(avg_RandomForest_exs))
print('mean squared error = {}'.format(avg_RandomForest_ms))
print('r2 = {}'.format(avg_RandomForest_r2))
print('mean squared log error = {}'.format(avg_RandomForest_msle))


print("Model 2: ExtraTreesRegressor ")
avg_ExtraTrees_exs = (sum_exs2)/5
avg_ExtraTrees_ms = (sum_ms2)/5
avg_ExtraTrees_r2 = (sum_r22)/5
avg_ExtraTrees_msle = (sum_msle2)/5

print('explained variance = {}'.format(avg_ExtraTrees_exs))
print('mean squared error = {}'.format(avg_ExtraTrees_ms))
print('r2 = {}'.format(avg_ExtraTrees_r2))
print('mean squared log error = {}'.format(avg_ExtraTrees_msle))


print("Model 3: DecisionTreeRegressor ")
avg_DecisionTree_exs = (sum_exs3)/5
avg_DecisionTree_ms = (sum_ms3)/5
avg_DecisionTree_r2 = (sum_r23)/5
avg_DecisionTree_msle = (sum_msle3)/5

print('explained variance = {}'.format(avg_DecisionTree_exs))
print('mean squared error = {}'.format(avg_DecisionTree_ms))
print('r2 = {}'.format(avg_DecisionTree_r2))
print('mean squared log error = {}'.format(avg_DecisionTree_msle))


print("Model 4: SupportVectorMachine ")
avg_SupportVector_exs = (sum_exs4)/5
avg_SupportVector_ms = (sum_ms4)/5
avg_SupportVector_r2 = (sum_r24)/5
avg_SupportVector_msle = (sum_msle4)/5

print('explained variance = {}'.format(avg_SupportVector_exs))
print('mean squared error = {}'.format(avg_SupportVector_ms))
print('r2 = {}'.format(avg_SupportVector_r2))
print('mean squared log error = {}'.format(avg_SupportVector_msle))

  
print("Model 5: AdaBoostRegressor ")
avg_AdaBoost_exs = (sum_exs5)/5
avg_AdaBoost_ms = (sum_ms5)/5
avg_AdaBoost_r2 = (sum_r25)/5
avg_AdaBoost_msle = (sum_msle5)/5

print('explained variance = {}'.format(avg_AdaBoost_exs))
print('mean squared error = {}'.format(avg_AdaBoost_ms))
print('r2 = {}'.format(avg_AdaBoost_r2))
print('mean squared log error = {}'.format(avg_AdaBoost_msle))  
  
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = features[['Temperature ','Concentration of carbon','Temperature anomaly ']]
corrmat = corrmat.corr()
f, ax = plt.subplots(figsize=(7,7))
sns.heatmap(corrmat, vmin=-.9, vmax=.9,annot=True,fmt=".2f",square=True, cmap = 'coolwarm')
plt.show() 


'''

import matplotlib.pyplot as plt
import seaborn as sns
corrmat = features[['Temperature ','Concentration of carbon','Temperature anomaly ']]
corrmat = corrmat.corr()
f, ax = plt.subplots(figsize=(7,7))
sns.heatmap(corrmat, vmin=-.9, vmax=.9,annot=True,fmt=".2f",square=True, cmap = 'coolwarm')
plt.show()

from sklearn import tree 
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,sealevels)
tree.export_graphviz(clf, out_file='tree.dot')    


'''




   
