# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 22:01:15 2025

@author: gutingyu
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib


alldata = pd.read_excel(r'your_path')
y = alldata['Nitrate']
X = alldata[['SST', 'SSS', 'MLD', 'Chla']]

x_train0, x_independence, y_train0, y_independence = train_test_split(X, y, test_size=0.1, random_state=41)
X_train, X_test, y_train, y_test = train_test_split(x_train0, y_train0, test_size=0.3, random_state=42)

rf1 = RandomForestRegressor(n_estimators=50, random_state=42)
rf2 = RandomForestRegressor(n_estimators=50, random_state=42)

# rf1
rf1.fit(x_train0, y_train0)
y_pred_train = rf1.predict(X_train)
y_pred_test = rf1.predict(X_test)

X_train_stacked = np.column_stack((X_train, y_pred_train))
X_test_stacked = np.column_stack((X_test, y_pred_test))

param_grid_rf1 = {
    'n_estimators': range(8, 29, 1),
    'max_depth': range(10, 25, 1),
    'min_samples_split': [0.5, 2, 3, 4, 5, 6, 7]
}

grid_search_rf1 = GridSearchCV(estimator=rf1, param_grid=param_grid_rf1, cv=5, scoring='neg_mean_squared_error')
grid_search_rf1.fit(x_train0, y_train0)
best_rf1 = grid_search_rf1.best_estimator_

best_rf1.fit(x_train0, y_train0)
y_pred_train_best = best_rf1.predict(X_train)
y_pred_test_best = best_rf1.predict(X_test)

X_train_stacked_best = np.column_stack((X_train, y_pred_train_best))
X_test_stacked_best = np.column_stack((X_test, y_pred_test_best))

param_grid_rf2 = {
    'n_estimators': range(1, 20, 1),
    'max_depth': range(5, 15, 1),
    'min_samples_split': [0.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

grid_search_rf2 = GridSearchCV(estimator=rf2, param_grid=param_grid_rf2, cv=5, scoring='neg_mean_squared_error')
grid_search_rf2.fit(np.column_stack((x_train0, best_rf1.predict(x_train0))), y_train0)
best_rf2 = grid_search_rf2.best_estimator_

best_rf2.fit(X_train_stacked_best, y_train)
y_pred_train_stacked_best = best_rf2.predict(X_train_stacked_best)
y_pred_test_stacked_best = best_rf2.predict(X_test_stacked_best)

joblib.dump((best_rf1, best_rf2), 'SRF.pkl')



import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.figure(figsize=(8, 6), dpi=1440)  
#plt.grid()
plt.plot(list(range(0,50,5)),list(range(0,50,5)),'k-', linewidth=1)
plt.gca().xaxis.set_major_locator(MultipleLocator(3))
plt.gca().yaxis.set_major_locator(MultipleLocator(3))
plt.hist2d(y_train,y_pred_train_stacked_best,bins=100,cmap='ocean_r',norm = Normalize(vmin=0, vmax=30))
#plt.hist2d(y_train,y_pred_train_stacked_best,bins=100,cmap='jet',norm = None)
# colorbar
#plt.xscale('log')
#plt.yscale('log')
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12.5)
cbar.ax.set_title('Number',fontsize=14)
# cbar.set_label('number',fontsize=14)
cbar.ax.tick_params(labelsize = 14)
plt.xlim(0,15)
plt.ylim(0,15)
plt.title('Model Training',fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Field Nitrate ($\\mu$mol/kg)',fontsize=16)
plt.ylabel('Modeled Nitrate ($\\mu$mol/kg)',fontsize=16)
plt.show()



import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16  
plt.figure(figsize=(8, 6), dpi=1440)  
plt.hist2d(y_test,y_pred_test_stacked_best,bins=100,cmap='ocean_r',norm = Normalize(vmin=0, vmax=20))
cbar=plt.colorbar() 
cbar.ax.set_title('Number',fontsize=14)
cbar.ax.tick_params(labelsize = 14)

plt.gca().set_xticks([0.01, 0.1,1,5,10,18])
plt.plot(list(range(0,50,5)),list(range(0,50,5)),'k-', linewidth=1)
plt.gca().xaxis.set_major_locator(MultipleLocator(3))
plt.gca().yaxis.set_major_locator(MultipleLocator(3))
plt.xlim(0,15)
plt.ylim(0,15)
plt.title('Model Testing',fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
#plt.gca().set_xscale('log')
#plt.gca().set_yscale('log')

plt.xlabel('Field Nitrate ($\\mu$mol/kg)',fontsize=16)
plt.ylabel('Modeled Nitrate ($\\mu$mol/kg)',fontsize=16)
plt.show()



import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16  
plt.figure(figsize=(8, 6), dpi=1440)  
plt.hist2d(y_independence,best_rf2.predict(np.column_stack((x_independence, best_rf1.predict(x_independence)))),bins=100,cmap='ocean_r',norm = Normalize(vmin=0, vmax=10))
cbar=plt.colorbar() 
cbar.ax.set_title('Number',fontsize=14)
cbar.ax.tick_params(labelsize = 14)
plt.xlim(0,15)
plt.ylim(0,15)
plt.plot(list(range(0,50,5)),list(range(0,50,5)),'k-', linewidth=1)
plt.gca().xaxis.set_major_locator(MultipleLocator(3))
plt.gca().yaxis.set_major_locator(MultipleLocator(3))
plt.title('Independent Validation',fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('Field Nitrate ($\\mu$mol/kg)',fontsize=16)
plt.ylabel('Modeled Nitrate ($\\mu$mol/kg)',fontsize=16)
plt.show()



        