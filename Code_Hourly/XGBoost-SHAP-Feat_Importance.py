# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:03:11 2023

@author: Paulo Rocha
"""


####################################################################################################################
# Libraries importing
####################################################################################################################

import time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

import seaborn as sns

import xgboost

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer

import shap


####################################################################################################################
# Overall settings
####################################################################################################################

plotXGBoost = wantPlot
persistRMSE = valid_persist_errors[0]


use_BoxCox = False

####################################################################################################################
# Train the XGBoost model and present its errors
####################################################################################################################
TOC = time.perf_counter() # start clock

# conversion
X_train = df_ML_train.dropna().iloc[:, 1:-1]
X_valid = df_ML_valid.dropna().iloc[:, 1:-1]
if use_BoxCox:
    pt = PowerTransformer()
    X_train = pt.fit_transform(X_train)
    X_valid = pt.transform(X_valid)

y_train = df_ML_train.dropna().iloc[:, -1]
y_valid = df_ML_valid.dropna().iloc[:, -1]

# Hyper parameters range intialization for tuning 

xgboost_reg = xgboost.XGBRegressor()

parameters={"learning_rate": (0.40, 0.60),
            "max_depth": [6, 10],
            "min_child_weight": [0.000001, 0.00001],
            "gamma":[0.000001, 0.00001],
            "colsample_bytree":[0.10, 0.50],
            "subsample":[0.25, 0.75]}

tuning_model=GridSearchCV(xgboost_reg,
                          param_grid=parameters,
                          scoring='neg_mean_squared_error',
                          #scoring='neg_mean_absolute_error',
                          refit=True,
                          cv=TimeSeriesSplit(n_splits=5),
                          #cv=[(slice(None), slice(None))],
                          n_jobs=-1,
                          verbose=0)

tuning_model.fit(X_train, y_train)

print('\nXGBoost regression model - best hyperparameters:')
print('V: ', tuning_model.best_params_)
print()

xgboost_reg = xgboost.XGBRegressor(**tuning_model.best_params_)

xgboost_reg.fit(X_train, y_train)

y_pred = xgboost_reg.predict(X_train)

train_XGB_error = calculate_errors(pd.DataFrame(y_pred).iloc[:,0], y_train, persistRMSE)
    
print(f"\nXGBoost Model RMSE for training = {train_XGB_error[0]:0.5f} ")
print(f"XGBoost Model nRMSE for training = {train_XGB_error[1]:0.5f} ")
print(f"XGBoost Model MAE for training = {train_XGB_error[2]:0.5f} ")
print(f"XGBoost Model nMAE for training = {train_XGB_error[3]:0.5f} ")
print(f"XGBoost Model MAPE for training = {train_XGB_error[4]:0.5f} ")
print(f"XGBoost Model MBE for training = {train_XGB_error[5]:0.5f} ")
print(f"XGBoost Model Skill for training = {train_XGB_error[6]:0.5f} \n")

y_pred = xgboost_reg.predict(X_valid)

valid_XGB_error = calculate_errors(pd.DataFrame(y_pred).iloc[:,0], y_valid, persistRMSE)
   
print(f"\nXGBoost Model RMSE for validation = {valid_XGB_error[0]:0.5f} ")
print(f"XGBoost Model nRMSE for validation = {valid_XGB_error[1]:0.5f} ")
print(f"XGBoost Model MAE for validation = {valid_XGB_error[2]:0.5f} ")
print(f"XGBoost Model nMAE for validation = {valid_XGB_error[3]:0.5f} ")
print(f"XGBoost Model MAPE for validation = {valid_XGB_error[4]:0.5f} ")
print(f"XGBoost Model MBE for validation = {valid_XGB_error[5]:0.5f} ")
print(f"XGBoost Model Skill for validation = {valid_XGB_error[6]:0.5f} \n")

plt.figure(figsize=(2.5*5,3.5*5))
xgboost.plot_importance(xgboost_reg, max_num_features=25)
#plt.title("xgboost.plot_importance(xgb_reg)")
plt.show()

shap.initjs()
#explainer = shap.Explainer(xgboost_reg, X_train) 
explainer = shap.TreeExplainer(xgboost_reg, X_train)#, feature_names=X_train.columns.tolist())

shap_values = explainer(X_valid) # must be run before each plot
plt.figure(figsize=(2.5*5,3.5*5))
shap.plots.beeswarm(shap_values, max_display=25, color=plt.get_cmap("winter_r"))

shap_values = explainer(X_valid) # must be run before each plot
shap.plots.bar(shap_values, max_display=25)

#shap_values = explainer(X_valid) # must be run before each plot
#shap.plots.heatmap(shap_values, max_display=25)

if plotXGBoost:
   
    ax = plt.figure(figsize=(3.5*5,3.5*5)).add_subplot(projection='3d')
    ax.plot(y_valid, y_pred, 'o', alpha=0.2,
            zs=0, zdir='z', label=None)
    ax.plot(np.array([np.min(y_valid), np.max(y_valid)]),
            np.array([np.min(y_valid), np.max(y_valid)]), color='purple', linewidth=4,
            zs=0, zdir='z', label=None)
    ax.plot(np.array([400, 400]),
            np.array([np.min(y_valid), np.max(y_valid)]), color = 'k', linestyle = ':', linewidth = 3,
            zs=0, zdir='z', label=None)
    ax.plot(np.array([500, 500]),
            np.array([np.min(y_valid), np.max(y_valid)]), color = 'k', linestyle = 'dashed', linewidth = 4,
            zs=0, zdir='z', label=None)
    ax.plot(np.array([np.min(y_valid), np.max(y_valid)]),
            np.array([400, 400]), color = 'k', linestyle = ':', linewidth = 3,
            zs=0, zdir='z', label=None)
    ax.plot(np.array([np.min(y_valid), np.max(y_valid)]),
            np.array([500, 500]), color = 'k', linestyle = 'dashed', linewidth = 4,
            zs=0, zdir='z', label=None)
    
    hist, bins = np.histogram(y_valid, bins=30)
    xs = (bins[:-1] + bins[1:])/2
    ax.bar(xs, hist, zs=np.max(y_valid), zdir='y', alpha=0.9, color='tab:green')
    hist, bins = np.histogram(y_pred, bins=30)
    xs = (bins[:-1] + bins[1:])/2
    ax.bar(xs, hist, zs=np.min(y_valid), zdir='x', alpha=0.9, color='midnightblue')
    
    # Make legend, set axes limits and labels
    #ax.legend()
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    #ax.set_zlim(0, 1)
    ax.set_xlabel('Measured', size=14)
    ax.set_ylabel('Forecasted', size=14)
    ax.set_zlabel('Frequency', size=14)
    
    ax.view_init(elev=30., azim=-45)
    plt.show()
    del ax, xs, bins, hist
    
    g = sns.JointGrid(height=10, 
                      ratio=2,
                      space=0.1,
                      marginal_ticks=True)
    x, y = y_valid, y_pred
    # add scatter plot layer
    #g.plot_joint(sns.regplot, order=2, scatter_kws={'alpha': 0.2})
    sns.regplot(x=x, y=y, order=1, color='midnightblue', ax=g.ax_joint, scatter_kws={'alpha': 0.2})
    g.ax_joint.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    g.ax_joint.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    g.ax_joint.axvline(x = 400, color = 'k', linestyle = ':', linewidth = 3)
    g.ax_joint.axvline(x = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    # add marginal density plot layer
    #g.plot_marginals(sns.histplot, kde=True)
    sns.histplot(x=x, bins=50, fill=True, color='tab:green', linewidth=1, ax=g.ax_marg_x, kde=True)
    sns.histplot(y=y, bins=50, fill=True, color='midnightblue', linewidth=1, ax=g.ax_marg_y, kde=True)   
    g.set_axis_labels('Measured', 'Forecasted')
    plt.show()
    
    plt.figure(figsize=(5.5*5,2.5*5))
    plt.plot(df_ML_valid.dropna().iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green', linewidth=2.5)
    plt.fill_between(df_ML_valid.dropna().iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    plt.plot(df_ML_valid.dropna().iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    plt.legend(shadow=True, fontsize=14)
    plt.gcf().autofmt_xdate()
    plt.title('XGBoost Regression - Validation Dataset - ' + str(STUDY_STATION) , size=20)
    plt.xlabel('Date', size=14)
    plt.ylabel('Discharge (m3/s)', size=14)
    #plt.ylim(74, 80)
    plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    plt.show()
    
    # plt.figure(figsize=(5.5*5,2.5*5))
    # plt.plot(df_ML_valid.dropna().iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green')
    # plt.fill_between(df_ML_valid.dropna().iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    # plt.plot(df_ML_valid.dropna().iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    # plt.gcf().autofmt_xdate()
    # plt.xlabel('Date', size=14)
    # plt.ylabel('Discharge (m3/s)', size=14)
    # plt.xlim([pd.to_datetime('2013-06-15', format = '%Y-%m-%d'),
    #          pd.to_datetime('2013-07-15', format = '%Y-%m-%d')])
    # #plt.xlim(df_ML_valid.dropna().iloc[0, 0]+datetime.timedelta(minutes=0.450*len(df_ML_valid.dropna())*10),
    # #         df_ML_valid.dropna().iloc[0, 0]+datetime.timedelta(minutes=0.550*len(df_ML_valid.dropna())*10))
    # #plt.ylim(74, 80)
    # plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    # plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    # plt.show()    

    # detail of the three peaks - peak2013
    plt.figure(figsize=(5.5*5,2.5*5))
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green')
    plt.fill_between(df_ML_valid.iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', size=14)
    plt.ylabel('Discharge (m3/s)', size=14)
    plt.xlim([pd.to_datetime('2013-06-15', format = '%Y-%m-%d'),
             pd.to_datetime('2013-07-15', format = '%Y-%m-%d')])
    #plt.ylim(74, 80)
    plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    plt.show()
    
    # detail of the three peaks - peak2012
    plt.figure(figsize=(5.5*5,2.5*5))
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green')
    plt.fill_between(df_ML_valid.iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', size=14)
    plt.ylabel('Discharge (m3/s)', size=14)
    plt.xlim([pd.to_datetime('2012-05-15', format = '%Y-%m-%d'),
             pd.to_datetime('2012-08-15', format = '%Y-%m-%d')])
    #plt.ylim(74, 80)
    plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    plt.show()
    
    # detail of the three peaks - peak2011
    plt.figure(figsize=(5.5*5,2.5*5))
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green')
    plt.fill_between(df_ML_valid.iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', size=14)
    plt.ylabel('Discharge (m3/s)', size=14)
    plt.xlim([pd.to_datetime('2011-05-15', format = '%Y-%m-%d'),
             pd.to_datetime('2011-08-15', format = '%Y-%m-%d')])
    #plt.ylim(74, 80)
    plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    plt.show()
    
    # deeper detail of the three peaks - peak2013
    plt.figure(figsize=(5.5*5,2.5*5))
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green')
    plt.fill_between(df_ML_valid.iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', size=14)
    plt.ylabel('Discharge (m3/s)', size=14)
    plt.xlim([pd.to_datetime('2013-06-19', format = '%Y-%m-%d'),
             pd.to_datetime('2013-06-23', format = '%Y-%m-%d')])
    #plt.ylim(74, 80)
    plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    plt.show()
    
    # deeper detail of the three peaks - peak2012
    plt.figure(figsize=(5.5*5,2.5*5))
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green')
    plt.fill_between(df_ML_valid.iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', size=14)
    plt.ylabel('Discharge (m3/s)', size=14)
    plt.xlim([pd.to_datetime('2012-06-01', format = '%Y-%m-%d'),
             pd.to_datetime('2012-07-01', format = '%Y-%m-%d')])
    #plt.ylim(74, 80)
    plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    plt.show()
    
    # deeper detail of the three peaks - peak2011
    plt.figure(figsize=(5.5*5,2.5*5))
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_valid, label='Measured Discharge', color='tab:green')
    plt.fill_between(df_ML_valid.iloc[:, 0], y_valid, color='tab:green', alpha=0.4)
    plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], y_pred, linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date', size=14)
    plt.ylabel('Discharge (m3/s)', size=14)
    plt.xlim([pd.to_datetime('2011-05-19', format = '%Y-%m-%d'),
             pd.to_datetime('2011-06-01', format = '%Y-%m-%d')])
    #plt.ylim(74, 80)
    plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
    plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
    plt.show()
  
TIC = time.perf_counter() # end clock
print(f"\nTrained the XGBoost model in {TIC - TOC:0.4f} seconds \n"); del TIC, TOC


####################################################################################################################
# THE END - After everything is running fine
####################################################################################################################
#import warnings
#warnings.filterwarnings("ignore")
# import sys
# sys.exit(0)


