# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:21:51 2023

@author: Paulo
"""

####################################################################################################################
# Libraries importing
####################################################################################################################

import os

import pandas as pd

import numpy as np

import torch

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

from torchmetrics import R2Score


####################################################################################################################
# Calculate error statistics
####################################################################################################################

def calculate_errors(predicted, target, persistRMSE=1):

    RMSE = np.sqrt(((predicted - target) ** 2).mean())
    
    nRMSE = RMSE/abs(target).mean()

    MAE = (abs(predicted - target)).mean()
    
    nMAE = MAE/abs(target).mean()
    
    try:
        MAPE = (abs((predicted - target)/target)).mean()
    except:
        MAPE = 10e7
    
    MBE = (predicted - target).mean()
    
    Skill = 1 - RMSE / persistRMSE
       
    r2score = R2Score()
    R2 = r2score(torch.squeeze(torch.tensor(predicted.values)), torch.squeeze(torch.tensor(target.values)))
       
    return RMSE, nRMSE, MAE, nMAE, MAPE, MBE, Skill, R2


####################################################################################################################
# Overall settings
####################################################################################################################

wantPlot = True

DATA_DIR = 'C:/Users/victo/Desktop/weblakes_papers/paper_bow/bow_river_data-20230428T184828Z-001/bow_river_data/'

STUDY_STATION = '05BH004'

TIME_RESOLUTION = 1 # in hours - not to be changed

TIMELAGS = 6
if TIMELAGS < 1: TIMELAGS = 1

TIMESHIFTS = 12 # timesteps to the future forecast - default = 1
if TIMESHIFTS < 1: TIMESHIFTS = 1

use_Itself_Data = True # False only works for ML models still

use_DoY_HoD = False

use_Precip_Data = False

use_SnowPillow_Data = False

use_Temperature_Data = False

YEAR_START = 2011
YEAR_END = 2022 #2022 # maximum to test train size convergence

LIST_DATASET_YEARS = []
for i in range(YEAR_START, YEAR_END+1):
    LIST_DATASET_YEARS.append(str(i))
del i

LIST_VALID_YEARS = [
                    '2011',
                    '2012',
                    '2013',     # biggest flood
                   ]

LIST_TRAIN_YEARS = [x for x in LIST_DATASET_YEARS if x not in LIST_VALID_YEARS]

del LIST_DATASET_YEARS


####################################################################################################################
# Read and process the main data
####################################################################################################################

# get a list of the .csv files
csv_discharge_files = [f for f in os.listdir(DATA_DIR + 'flow/') if (f.endswith('.csv'))]
csv_precip_files = [f for f in os.listdir(DATA_DIR + 'precipitation/') if (f.endswith('.csv'))]
csv_snow_pillow_files = [f for f in os.listdir(DATA_DIR + 'snow_pillow/') if (f.endswith('.csv'))]
csv_temperature_files = [f for f in os.listdir(DATA_DIR + 'temperature/') if (f.endswith('.csv'))]

# reorder by the distance from the study station - not ended yet
for station in csv_discharge_files:
    if STUDY_STATION in station:
        csv_discharge_files.remove(station)
        csv_discharge_files = [station] + csv_discharge_files
# to be completed if needed #########################

# build list of dataframes - one per station

list_df_GNN_flow = []
list_df_GNN_flow_train = []
list_df_GNN_flow_valid = []

for station_idx in range(0,len(csv_discharge_files)):
    print(csv_discharge_files[station_idx])
    list_df_GNN_flow.append(pd.read_csv(DATA_DIR + 'flow/' + csv_discharge_files[station_idx], skiprows=23, usecols=[0, 1, 2], encoding = 'unicode_escape'))
    
    list_df_GNN_flow[station_idx].columns.values[2] = "Flow"
    list_df_GNN_flow[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_flow[station_idx]['Date'] + ' ' + list_df_GNN_flow[station_idx]['Time'])
    list_df_GNN_flow[station_idx][csv_discharge_files[station_idx].split('.')[0][14:21]] = list_df_GNN_flow[station_idx]['Flow']
    
    list_df_GNN_flow[station_idx].drop(columns=['Date', 'Time', 'Flow'], inplace=True)
    
    list_df_GNN_flow[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_flow[station_idx]['Timestamp'])
    
    list_df_GNN_flow[station_idx].columns.values[1] = list_df_GNN_flow[station_idx].columns.values[1] + '_Flow'

    print('Before:')
    print(min(list_df_GNN_flow[station_idx]['Timestamp']))
    print(max(list_df_GNN_flow[station_idx]['Timestamp']))
    print()
    
    list_df_GNN_flow[station_idx] = list_df_GNN_flow[station_idx][list_df_GNN_flow[station_idx]['Timestamp'] >= pd.to_datetime(str(YEAR_START) + '-01-01')]
    list_df_GNN_flow[station_idx] = list_df_GNN_flow[station_idx][list_df_GNN_flow[station_idx]['Timestamp'] <= pd.to_datetime(str(YEAR_END+1) + '-01-01')]
    list_df_GNN_flow[station_idx].reset_index(drop=True, inplace=True)
    
    print('After:')
    print(min(list_df_GNN_flow[station_idx]['Timestamp']))
    print(max(list_df_GNN_flow[station_idx]['Timestamp']))
    print(len(list_df_GNN_flow[station_idx]['Timestamp']))

    print()

    list_df_GNN_flow[station_idx].set_index(['Timestamp'], inplace=True)
    list_df_GNN_flow[station_idx] = list_df_GNN_flow[station_idx].resample('1H').mean()
    list_df_GNN_flow[station_idx].reset_index(drop=False, inplace=True)
  
    # apply lags by shifting dataframe column down
    for lag in range(TIMELAGS):
        columnName = list_df_GNN_flow[station_idx].columns.values[1] + ' _ lag(' + str(lag+1) + ')'
        list_df_GNN_flow[station_idx][columnName] = list_df_GNN_flow[station_idx].iloc[:, 1].shift(lag+1)

    # conciliation with the study station dataset - BE CAREFUL WITH THE ORIGINAL SIZE OF THE DATASET !!!
    if station_idx != 0:
        list_df_GNN_flow[station_idx] = pd.merge(list_df_GNN_flow[0], list_df_GNN_flow[station_idx],
                          how='left', on=['Timestamp'])
        for i in range(0, TIMELAGS+1): list_df_GNN_flow[station_idx].drop(columns=[list_df_GNN_flow[station_idx].columns.values[1]], inplace=True)

    # put 'Measured Value' (TIME SHIFTED) column at the end
    columnName = list_df_GNN_flow[station_idx].columns.values[1] + ' - ' + str(TIMESHIFTS) + '-hour'
    list_df_GNN_flow[station_idx][columnName] = list_df_GNN_flow[station_idx].iloc[:, 1].shift(-(TIMESHIFTS-1))
    list_df_GNN_flow[station_idx].drop(columns=[list_df_GNN_flow[station_idx].columns.values[1]], inplace=True)
    list_df_GNN_flow[station_idx].fillna(method="ffill", inplace=True)
    list_df_GNN_flow[station_idx].fillna(method="bfill", inplace=True)
    
    # split train / valid sets
    list_df_GNN_flow_train.append(list_df_GNN_flow[station_idx][pd.to_datetime(list_df_GNN_flow[0]['Timestamp']).dt.year.astype(str).isin(LIST_TRAIN_YEARS)])
    list_df_GNN_flow_valid.append(list_df_GNN_flow[station_idx][pd.to_datetime(list_df_GNN_flow[0]['Timestamp']).dt.year.astype(str).isin(LIST_VALID_YEARS)])
  
    if wantPlot:
        plt.figure(figsize=(5.5*5,2.5*5))
        plt.plot(list_df_GNN_flow[station_idx]['Timestamp'], list_df_GNN_flow[station_idx].iloc[:, -1], linewidth=2.5, label='Measured Discharge (m3/s)', color='tab:blue')
        plt.fill_between(list_df_GNN_flow[station_idx]['Timestamp'], list_df_GNN_flow[station_idx].iloc[:, -1], color='tab:blue', alpha=0.4)
        plt.legend(shadow=True, fontsize=14)
        plt.gcf().autofmt_xdate()
        plt.title('Full Dataset - Discharge Station ' + csv_discharge_files[station_idx].split('.')[0][14:21], size=20)
        plt.xlabel('Timestamp', size=14)
        plt.ylabel('Discharge (m3/s)', size=14)
        plt.axvline(x = list_df_GNN_flow_train[station_idx]['Timestamp'].iloc[0], color = 'k', linestyle = 'dashed', linewidth = 4)
        plt.show()

list_df_GNN_precip = []
list_df_GNN_precip_train = []
list_df_GNN_precip_valid = []

if use_Precip_Data:

    for station_idx in range(0,len(csv_precip_files)):
        print(csv_precip_files[station_idx])
        list_df_GNN_precip.append(pd.read_csv(DATA_DIR + 'precipitation/' + csv_precip_files[station_idx], skiprows=23, usecols=[0, 1, 2]))
        
        list_df_GNN_precip[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_precip[station_idx]['Date'] + ' ' + list_df_GNN_precip[station_idx]['Time'])
        list_df_GNN_precip[station_idx][csv_precip_files[station_idx][14:-22]] = list_df_GNN_precip[station_idx]['Value(mm)']
        
        list_df_GNN_precip[station_idx].drop(columns=['Date', 'Time', 'Value(mm)'], inplace=True)
        
        list_df_GNN_precip[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_precip[station_idx]['Timestamp'])
        
        list_df_GNN_precip[station_idx].columns.values[1] = csv_precip_files[station_idx][14:-22] + '_Precip'

        print('Before:')
        print(min(list_df_GNN_precip[station_idx]['Timestamp']))
        print(max(list_df_GNN_precip[station_idx]['Timestamp']))
        print()
        
        list_df_GNN_precip[station_idx] = list_df_GNN_precip[station_idx][list_df_GNN_precip[station_idx]['Timestamp'] >= pd.to_datetime(str(YEAR_START) + '-01-01')]
        list_df_GNN_precip[station_idx] = list_df_GNN_precip[station_idx][list_df_GNN_precip[station_idx]['Timestamp'] <= pd.to_datetime(str(YEAR_END+1) + '-01-01')]
        list_df_GNN_precip[station_idx].reset_index(drop=True, inplace=True)
        
        print('After:')
        print(min(list_df_GNN_precip[station_idx]['Timestamp']))
        print(max(list_df_GNN_precip[station_idx]['Timestamp']))
        print(len(list_df_GNN_precip[station_idx]['Timestamp']))

        print()

        list_df_GNN_precip[station_idx].set_index(['Timestamp'], inplace=True)
        list_df_GNN_precip[station_idx] = list_df_GNN_precip[station_idx].resample('1H').sum()
        list_df_GNN_precip[station_idx].reset_index(drop=False, inplace=True)
      
        # apply lags by shifting dataframe column down
        for lag in range(TIMELAGS):
            columnName = list_df_GNN_precip[station_idx].columns.values[1] + ' _ lag(' + str(lag+1) + ')'
            list_df_GNN_precip[station_idx][columnName] = list_df_GNN_precip[station_idx].iloc[:, 1].shift(lag+1)

        # conciliation with the study station dataset - BE CAREFUL WITH THE ORIGINAL SIZE OF THE DATASET !!!
        list_df_GNN_precip[station_idx] = pd.merge(list_df_GNN_flow[0], list_df_GNN_precip[station_idx],
                          how='left', on=['Timestamp'])
        for i in range(0, TIMELAGS+1): list_df_GNN_precip[station_idx].drop(columns=[list_df_GNN_precip[station_idx].columns.values[1]], inplace=True)

        # put 'Measured Value' (TIME SHIFTED) column at the end
        columnName = list_df_GNN_precip[station_idx].columns.values[1] + ' - ' + str(TIMESHIFTS) + '-hour'
        list_df_GNN_precip[station_idx][columnName] = list_df_GNN_precip[station_idx].iloc[:, 1].shift(-(TIMESHIFTS-1))
        list_df_GNN_precip[station_idx].drop(columns=[list_df_GNN_precip[station_idx].columns.values[1]], inplace=True)
        list_df_GNN_precip[station_idx].fillna(method="ffill", inplace=True)
        list_df_GNN_precip[station_idx].fillna(method="bfill", inplace=True)
        
        # split train / valid sets
        list_df_GNN_precip_train.append(list_df_GNN_precip[station_idx][pd.to_datetime(list_df_GNN_precip[0]['Timestamp']).dt.year.astype(str).isin(LIST_TRAIN_YEARS)])
        list_df_GNN_precip_valid.append(list_df_GNN_precip[station_idx][pd.to_datetime(list_df_GNN_precip[0]['Timestamp']).dt.year.astype(str).isin(LIST_VALID_YEARS)])
      
        if wantPlot:
            plt.figure(figsize=(5.5*5,2.5*5))
            plt.plot(list_df_GNN_precip[station_idx]['Timestamp'], list_df_GNN_precip[station_idx].iloc[:, -1], linewidth=2.5, label='Measured Precipitation (mm)', color='tab:green')
            plt.fill_between(list_df_GNN_precip[station_idx]['Timestamp'], list_df_GNN_precip[station_idx].iloc[:, -1], color='tab:green', alpha=0.4)
            plt.legend(shadow=True, fontsize=14)
            plt.gcf().autofmt_xdate()
            plt.title('Full Dataset - Precipitation Station ' + csv_precip_files[station_idx][14:-22], size=20)
            plt.xlabel('Timestamp', size=14)
            plt.ylabel('Precipitation (mm)', size=14)
            plt.axvline(x = list_df_GNN_precip_train[station_idx]['Timestamp'].iloc[0], color = 'k', linestyle = 'dashed', linewidth = 4)
            plt.show()
    
list_df_GNN_snowpillow = []
list_df_GNN_snowpillow_train = []
list_df_GNN_snowpillow_valid = []

if use_SnowPillow_Data:

    for station_idx in range(0,len(csv_snow_pillow_files)):
        print(csv_snow_pillow_files[station_idx])
        list_df_GNN_snowpillow.append(pd.read_csv(DATA_DIR + 'snow_pillow/' + csv_snow_pillow_files[station_idx], skiprows=23, usecols=[0, 1, 2]))
        
        list_df_GNN_snowpillow[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_snowpillow[station_idx]['Date'] + ' ' + list_df_GNN_snowpillow[station_idx]['Time'])
        list_df_GNN_snowpillow[station_idx][csv_snow_pillow_files[station_idx][14:-22]] = list_df_GNN_snowpillow[station_idx]['Value(mm)']
        
        list_df_GNN_snowpillow[station_idx].drop(columns=['Date', 'Time', 'Value(mm)'], inplace=True)
        
        list_df_GNN_snowpillow[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_snowpillow[station_idx]['Timestamp'])
        
        list_df_GNN_snowpillow[station_idx].columns.values[1] = csv_snow_pillow_files[station_idx][14:-22] + '_SnowPillow'

        print('Before:')
        print(min(list_df_GNN_snowpillow[station_idx]['Timestamp']))
        print(max(list_df_GNN_snowpillow[station_idx]['Timestamp']))
        print()
        
        list_df_GNN_snowpillow[station_idx] = list_df_GNN_snowpillow[station_idx][list_df_GNN_snowpillow[station_idx]['Timestamp'] >= pd.to_datetime(str(YEAR_START) + '-01-01')]
        list_df_GNN_snowpillow[station_idx] = list_df_GNN_snowpillow[station_idx][list_df_GNN_snowpillow[station_idx]['Timestamp'] <= pd.to_datetime(str(YEAR_END+1) + '-01-01')]
        list_df_GNN_snowpillow[station_idx].reset_index(drop=True, inplace=True)
        
        print('After:')
        print(min(list_df_GNN_snowpillow[station_idx]['Timestamp']))
        print(max(list_df_GNN_snowpillow[station_idx]['Timestamp']))
        print(len(list_df_GNN_snowpillow[station_idx]['Timestamp']))

        print()

        list_df_GNN_snowpillow[station_idx].set_index(['Timestamp'], inplace=True)
        list_df_GNN_snowpillow[station_idx] = list_df_GNN_snowpillow[station_idx].resample('1H').mean()
        list_df_GNN_snowpillow[station_idx].reset_index(drop=False, inplace=True)
      
        # apply lags by shifting dataframe column down
        for lag in range(TIMELAGS):
            columnName = list_df_GNN_snowpillow[station_idx].columns.values[1] + ' _ lag(' + str(lag+1) + ')'
            list_df_GNN_snowpillow[station_idx][columnName] = list_df_GNN_snowpillow[station_idx].iloc[:, 1].shift(lag+1)

        # conciliation with the study station dataset - BE CAREFUL WITH THE ORIGINAL SIZE OF THE DATASET !!!
        list_df_GNN_snowpillow[station_idx] = pd.merge(list_df_GNN_flow[0], list_df_GNN_snowpillow[station_idx],
                          how='left', on=['Timestamp'])
        for i in range(0, TIMELAGS+1): list_df_GNN_snowpillow[station_idx].drop(columns=[list_df_GNN_snowpillow[station_idx].columns.values[1]], inplace=True)

        # put 'Measured Value' (TIME SHIFTED) column at the end
        columnName = list_df_GNN_snowpillow[station_idx].columns.values[1] + ' - ' + str(TIMESHIFTS) + '-hour'
        list_df_GNN_snowpillow[station_idx][columnName] = list_df_GNN_snowpillow[station_idx].iloc[:, 1].shift(-(TIMESHIFTS-1))
        list_df_GNN_snowpillow[station_idx].drop(columns=[list_df_GNN_snowpillow[station_idx].columns.values[1]], inplace=True)
        list_df_GNN_snowpillow[station_idx].fillna(method="ffill", inplace=True)
        list_df_GNN_snowpillow[station_idx].fillna(method="bfill", inplace=True)
        
        # split train / valid sets
        list_df_GNN_snowpillow_train.append(list_df_GNN_snowpillow[station_idx][pd.to_datetime(list_df_GNN_snowpillow[0]['Timestamp']).dt.year.astype(str).isin(LIST_TRAIN_YEARS)])
        list_df_GNN_snowpillow_valid.append(list_df_GNN_snowpillow[station_idx][pd.to_datetime(list_df_GNN_snowpillow[0]['Timestamp']).dt.year.astype(str).isin(LIST_VALID_YEARS)])
      
        if wantPlot:
            plt.figure(figsize=(5.5*5,2.5*5))
            plt.plot(list_df_GNN_snowpillow[station_idx]['Timestamp'], list_df_GNN_snowpillow[station_idx].iloc[:, -1], linewidth=2.5, label='Measured Snow Pillow Height (mm)', color='tab:gray')
            plt.fill_between(list_df_GNN_snowpillow[station_idx]['Timestamp'], list_df_GNN_snowpillow[station_idx].iloc[:, -1], color='tab:gray', alpha=0.4)
            plt.legend(shadow=True, fontsize=14)
            plt.gcf().autofmt_xdate()
            plt.title('Full Dataset - Snow Pillow Station ' + csv_snow_pillow_files[station_idx][14:-22], size=20)
            plt.xlabel('Timestamp', size=14)
            plt.ylabel('Snow Pillow (mm)', size=14)
            plt.axvline(x = list_df_GNN_snowpillow_train[station_idx]['Timestamp'].iloc[0], color = 'k', linestyle = 'dashed', linewidth = 4)
            plt.show()

list_df_GNN_temperature = []
list_df_GNN_temperature_train = []
list_df_GNN_temperature_valid = []

if use_Temperature_Data:

    for station_idx in range(0,len(csv_temperature_files)):
        print(csv_temperature_files[station_idx])
        list_df_GNN_temperature.append(pd.read_csv(DATA_DIR + 'temperature/' + csv_temperature_files[station_idx], usecols=[0, 1, 2], encoding = 'unicode_escape'))
        
        list_df_GNN_temperature[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_temperature[station_idx].iloc[:, 1].copy(deep="all"))
        list_df_GNN_temperature[station_idx]['Temperature'] = list_df_GNN_temperature[station_idx].iloc[:, 2].copy(deep="all")
        list_df_GNN_temperature[station_idx].drop(list_df_GNN_temperature[station_idx].columns[[1, 2]], axis=1, inplace=True)
        
    df_tmp = pd.concat(list_df_GNN_temperature)
    list_df_GNN_temperature = []
    
    list_temperature_stations = df_tmp['Station Name'].unique()
    list_purge_temperature_stations = [
                                       #'Banff CS',
                                       #'Bow Valley',
                                       #'Cop Upper',
                                       #'Jumpingpound Ranger Station',
                                       #'Pika Run',
                                       #'South Ghost Headwaters',
                                       'Calgary Springbank A',   # always to be purged - lack of data
                                       ]
    list_temperature_stations = [ele for ele in list(list_temperature_stations) if ele not in list_purge_temperature_stations]
    
    for station in list_temperature_stations:
        list_df_GNN_temperature.append(df_tmp[df_tmp['Station Name']==station].copy(deep="all"))
    del df_tmp, station

    for station_idx in range(0,len(list_df_GNN_temperature)): 
        list_df_GNN_temperature[station_idx].columns.values[2] = str(list_df_GNN_temperature[station_idx].iloc[:, 0].unique())[2:-2] + '_Temp'
        list_df_GNN_temperature[station_idx].drop(columns=[list_df_GNN_temperature[station_idx].columns.values[0]], inplace=True)
        list_df_GNN_temperature[station_idx]['Timestamp'] = pd.to_datetime(list_df_GNN_temperature[station_idx]['Timestamp'].copy(deep="all"))

        print('Before:')
        print(min(list_df_GNN_temperature[station_idx]['Timestamp']))
        print(max(list_df_GNN_temperature[station_idx]['Timestamp']))
        print()
        
        list_df_GNN_temperature[station_idx] = list_df_GNN_temperature[station_idx][list_df_GNN_temperature[station_idx]['Timestamp'] >= pd.to_datetime(str(YEAR_START) + '-01-01')]
        list_df_GNN_temperature[station_idx] = list_df_GNN_temperature[station_idx][list_df_GNN_temperature[station_idx]['Timestamp'] <= pd.to_datetime(str(YEAR_END+1) + '-01-01')]
        list_df_GNN_temperature[station_idx].reset_index(drop=True, inplace=True)
        
        print('After:')
        print(min(list_df_GNN_temperature[station_idx]['Timestamp']))
        print(max(list_df_GNN_temperature[station_idx]['Timestamp']))
        print(len(list_df_GNN_temperature[station_idx]['Timestamp']))

        print()

        list_df_GNN_temperature[station_idx].set_index(['Timestamp'], inplace=True)
        list_df_GNN_temperature[station_idx] = list_df_GNN_temperature[station_idx].resample('1H').mean().copy(deep="all")
        list_df_GNN_temperature[station_idx].reset_index(drop=False, inplace=True)
      
        # apply lags by shifting dataframe column down
        for lag in range(TIMELAGS):
            columnName = list_df_GNN_temperature[station_idx].columns.values[1] + ' _ lag(' + str(lag+1) + ')'
            list_df_GNN_temperature[station_idx][columnName] = list_df_GNN_temperature[station_idx].iloc[:, 1].shift(lag+1)

        # conciliation with the study station dataset - BE CAREFUL WITH THE ORIGINAL SIZE OF THE DATASET !!!
        list_df_GNN_temperature[station_idx] = pd.merge(list_df_GNN_flow[0], list_df_GNN_temperature[station_idx],
                          how='left', on=['Timestamp'])
        for i in range(0, TIMELAGS+1): list_df_GNN_temperature[station_idx].drop(columns=[list_df_GNN_temperature[station_idx].columns.values[1]], inplace=True)

        # put 'Measured Value' (TIME SHIFTED) column at the end
        columnName = list_df_GNN_temperature[station_idx].columns.values[1] + ' - ' + str(TIMESHIFTS) + '-hour'
        list_df_GNN_temperature[station_idx][columnName] = list_df_GNN_temperature[station_idx].iloc[:, 1].shift(-(TIMESHIFTS-1))
        list_df_GNN_temperature[station_idx].drop(columns=[list_df_GNN_temperature[station_idx].columns.values[1]], inplace=True)
        list_df_GNN_temperature[station_idx].fillna(method="ffill", inplace=True)
        list_df_GNN_temperature[station_idx].fillna(method="bfill", inplace=True)
        
        # split train / valid sets
        list_df_GNN_temperature_train.append(list_df_GNN_temperature[station_idx][pd.to_datetime(list_df_GNN_temperature[0]['Timestamp']).dt.year.astype(str).isin(LIST_TRAIN_YEARS)])
        list_df_GNN_temperature_valid.append(list_df_GNN_temperature[station_idx][pd.to_datetime(list_df_GNN_temperature[0]['Timestamp']).dt.year.astype(str).isin(LIST_VALID_YEARS)])
      
        if wantPlot:
            plt.figure(figsize=(5.5*5,2.5*5))
            plt.plot(list_df_GNN_temperature[station_idx]['Timestamp'], list_df_GNN_temperature[station_idx].iloc[:, -1], linewidth=2.5, label='Measured Temperature (oC)', color='gold')
            plt.fill_between(list_df_GNN_temperature[station_idx]['Timestamp'], list_df_GNN_temperature[station_idx].iloc[:, -1], color='gold', alpha=0.4)
            plt.legend(shadow=True, fontsize=14)
            plt.gcf().autofmt_xdate()
            plt.title('Full Dataset - Temperature Station: ' + list_df_GNN_temperature[station_idx].columns.values[1][:-14], size=20)
            plt.xlabel('Timestamp', size=14)
            plt.ylabel('Temperature (oC)', size=14)
            plt.axvline(x = list_df_GNN_temperature_train[station_idx]['Timestamp'].iloc[0], color = 'k', linestyle = 'dashed', linewidth = 4)
            plt.show()
    
del station_idx


####################################################################################################################
# Join the dataframes back - for ML models
####################################################################################################################

df_ML_train = list_df_GNN_flow_train[0].iloc[:, :(1+use_Itself_Data*TIMELAGS)].copy(deep="all")

if len(list_df_GNN_flow_train) > 1:
    for df_idx in range(1, len(list_df_GNN_flow_train)):
        df_ML_train = pd.merge(df_ML_train, list_df_GNN_flow_train[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])

if use_Precip_Data:
    for df_idx in range(0, len(list_df_GNN_precip_train)):
        df_ML_train = pd.merge(df_ML_train, list_df_GNN_precip_train[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])
        
if use_SnowPillow_Data:
    for df_idx in range(0, len(list_df_GNN_snowpillow_train)):
        df_ML_train = pd.merge(df_ML_train, list_df_GNN_snowpillow_train[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])
        
if use_Temperature_Data:
    for df_idx in range(0, len(list_df_GNN_temperature_train)):
        df_ML_train = pd.merge(df_ML_train, list_df_GNN_temperature_train[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])

df_ML_train = pd.merge(df_ML_train, list_df_GNN_flow_train[0].iloc[:, [0,-1]],
                 on=['Timestamp'])

df_ML_valid = list_df_GNN_flow_valid[0].iloc[:, :(1+use_Itself_Data*TIMELAGS)].copy(deep="all")

if len(list_df_GNN_flow_valid) > 1:
    for df_idx in range(1, len(list_df_GNN_flow_valid)):
        df_ML_valid = pd.merge(df_ML_valid, list_df_GNN_flow_valid[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])

if use_Precip_Data:
    for df_idx in range(0, len(list_df_GNN_precip_valid)):
        df_ML_valid = pd.merge(df_ML_valid, list_df_GNN_precip_valid[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])
        
if use_SnowPillow_Data:
    for df_idx in range(0, len(list_df_GNN_snowpillow_valid)):
        df_ML_valid = pd.merge(df_ML_valid, list_df_GNN_snowpillow_valid[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])
        
if use_Temperature_Data:
    for df_idx in range(0, len(list_df_GNN_temperature_valid)):
        df_ML_valid = pd.merge(df_ML_valid, list_df_GNN_temperature_valid[df_idx].iloc[:, :-1],
                         how='left', on=['Timestamp'])

df_ML_valid = pd.merge(df_ML_valid, list_df_GNN_flow_valid[0].iloc[:, [0,-1]],
                 on=['Timestamp'])

if use_DoY_HoD:
    df_ML_train.insert(1, 'sin(DoY)', (np.sin(2*np.pi*pd.to_datetime(df_ML_train['Timestamp']).dt.dayofyear/365)))
    df_ML_train.insert(2, 'cos(DoY)', (np.cos(2*np.pi*pd.to_datetime(df_ML_train['Timestamp']).dt.dayofyear/365)))
    df_ML_train.insert(3, 'sin(HoD)', (np.sin(2*np.pi*pd.to_datetime(df_ML_train['Timestamp']).dt.hour/24)))
    df_ML_train.insert(4, 'cos(HoD)', (np.cos(2*np.pi*pd.to_datetime(df_ML_train['Timestamp']).dt.hour/24)))
    
    df_ML_valid.insert(1, 'sin(DoY)', (np.sin(2*np.pi*pd.to_datetime(df_ML_valid['Timestamp']).dt.dayofyear/365)))
    df_ML_valid.insert(2, 'cos(DoY)', (np.cos(2*np.pi*pd.to_datetime(df_ML_valid['Timestamp']).dt.dayofyear/365)))
    df_ML_valid.insert(3, 'sin(HoD)', (np.sin(2*np.pi*pd.to_datetime(df_ML_valid['Timestamp']).dt.hour/24)))
    df_ML_valid.insert(4, 'cos(HoD)', (np.cos(2*np.pi*pd.to_datetime(df_ML_valid['Timestamp']).dt.hour/24)))
    
    # for df_idx in range(0, len(list_df_GNN)):
    #     list_df_GNN[df_idx].insert(1, 'sin(DoY)', (np.sin(2*np.pi*df_ML['timestamp'].dt.dayofyear/365)))
    #     list_df_GNN[df_idx].insert(2, 'cos(DoY)', (np.cos(2*np.pi*df_ML['timestamp'].dt.dayofyear/365)))
    #     list_df_GNN[df_idx].insert(3, 'sin(HoD)', (np.sin(2*np.pi*df_ML['timestamp'].dt.hour/24)))
    #     list_df_GNN[df_idx].insert(4, 'cos(HoD)', (np.cos(2*np.pi*df_ML['timestamp'].dt.hour/24)))
    # del df_idx

try:
    del i, lag, df_idx, columnName, list_df_GNN_flow, list_df_GNN_precip, list_df_GNN_snowpillow, list_df_GNN_temperature, list_temperature_stations, list_purge_temperature_stations
except:
    pass

####################################################################################################################
# Calculate persistence errors
####################################################################################################################

train_persist_errors = calculate_errors(df_ML_train.iloc[:, -1].shift(-TIMESHIFTS).fillna(method="ffill"),
                                        df_ML_train.iloc[:, -1])

valid_persist_errors = calculate_errors(df_ML_valid.iloc[:, -1].shift(-TIMESHIFTS).fillna(method="ffill"),
                                        df_ML_valid.iloc[:, -1])

print('#########################################################################################################')
print('# Persistence Model')
print('#########################################################################################################')

print(f"\nPersistence Model RMSE for training = {train_persist_errors[0]:0.5f} ")
print(f"Persistence Model nRMSE for training = {train_persist_errors[1]:0.5f} ")
print(f"Persistence Model MAE for training = {train_persist_errors[2]:0.5f} ")
print(f"Persistence Model nMAE for training = {train_persist_errors[3]:0.5f} ")
print(f"Persistence Model MAPE for training = {train_persist_errors[4]:0.5f} ")
print(f"Persistence Model MBE for training = {train_persist_errors[5]:0.5f} ")
print(f"Persistence Model R2 for training = {train_persist_errors[7]:0.5f} \n")

print(f"\nPersistence Model RMSE for validation = {valid_persist_errors[0]:0.5f} ")
print(f"Persistence Model nRMSE for validation = {valid_persist_errors[1]:0.5f} ")
print(f"Persistence Model MAE for validation = {valid_persist_errors[2]:0.5f} ")
print(f"Persistence Model nMAE for validation = {valid_persist_errors[3]:0.5f} ")
print(f"Persistence Model MAPE for validation = {valid_persist_errors[4]:0.5f} ")
print(f"Persistence Model MBE for validation = {valid_persist_errors[5]:0.5f} ")
print(f"Persistence Model R2 for validation = {valid_persist_errors[7]:0.5f} \n")


