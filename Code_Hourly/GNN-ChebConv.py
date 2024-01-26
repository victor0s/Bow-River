# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:44:18 2022

@author: Paulo
"""

####################################################################################################################
# Libraries importing
####################################################################################################################

import torch
print('\nTorch version: ', torch.__version__)
print('CUDA available:', torch.cuda.is_available(), '\n') 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2023)

import datetime
import time

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

import seaborn as sns

from torchmetrics import R2Score

#from torch import nn

import torch.nn.functional as F
from torch_geometric.nn import ChebConv


####################################################################################################################
# Overall settings
####################################################################################################################

EPOCHS = 350
LEARNING_RATE = 1e-4
#LEARNING_RATE = 5e-5

target_mean_P = np.mean(list_df_GNN_train[0].iloc[:, -1], axis=0)
target_stdev_P = np.std(list_df_GNN_train[0].iloc[:, -1], axis=0)

plotChebConv = wantPlot


####################################################################################################################
# To release GPU memory space
####################################################################################################################

def cleans_GPU():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

cleans_GPU()


####################################################################################################################
# ChebConv GNN structure
####################################################################################################################

class ChebConvGNN(torch.nn.Module):
    def __init__(self, num_feat, num_nodes, device='cpu'):
        super().__init__()
        self.num_feat = num_feat
        self.num_nodes = num_nodes
        self.hid = 128
        self.K = 2
        self.conv1 = ChebConv(num_feat, self.hid, K=self.K)
        self.conv2 = ChebConv(self.hid, self.hid, K=self.K)
        self.conv3 = ChebConv(self.hid, self.hid, K=self.K)
        self.conv4 = ChebConv(self.hid, self.hid, K=self.K)
        self.lin1 = torch.nn.Linear(self.hid, self.hid)
        self.lin2 = torch.nn.Linear(self.hid, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print(x.shape, 'original')

        x = self.conv1(x, edge_index)
        #print(x.shape, 'after conv1')
        
        x = F.leaky_relu(x, 0.1)
        #print(x.shape, 'after relu')
        
        x = F.dropout(x, p=0.10, training=self.training)
        #print(x.shape, 'after dropout')
        
        x = self.conv2(x, edge_index)
        #print(x.shape, 'after conv2')
        
        x = F.leaky_relu(x, 0.1)
        #print(x.shape, 'after relu')
        
        x = F.dropout(x, p=0.10, training=self.training)
        #print(x.shape, 'after dropout')
        
        x = self.conv3(x, edge_index)
        #print(x.shape, 'after conv3')
        
        x = F.leaky_relu(x, 0.1)
        #print(x.shape, 'after relu')
        
        x = F.dropout(x, p=0.10, training=self.training)
        #print(x.shape, 'after dropout')
        
        x = self.conv4(x, edge_index)
        #print(x.shape, 'after conv4')
        
        x = self.lin1(x)
        #print(x)
        #print(x.shape, 'after lin1')
        
        x = F.leaky_relu(x, 0.1)
        #print(x.shape, 'after relu')
        
        x = self.lin2(x)
        #print(x)
        #print(x.shape, 'after lin2')
        
        # get the node 0 (station under study) results only
        x_0 = x[range(0, len(x), self.num_nodes)]
        #print(x_0)
        #print(x_0.shape)

        return x_0 #F.tanh(x_0)


####################################################################################################################
# Define the network
####################################################################################################################
TOC = time.perf_counter() # start clock

model = ChebConvGNN(data_list[0].x.shape[1], data_list[0].x.shape[0]).to(DEVICE)

print()

print('MODEL STRUCTURE:\n')
print(model)

TIC = time.perf_counter() # end clock
print(f"\nCreated and sent the ChebConv model to {torch.cuda.get_device_name(DEVICE).upper()} in {TIC - TOC:0.4f} seconds \n")
time.sleep(7)


####################################################################################################################
# Defines the model evaluation
####################################################################################################################

def evaluate_model(valid_dl, model, DEVICE='cpu', persistRMSE=1.0, plot=False, evalEPA=False):
    model.eval()
    predictions, actuals = list(), list()
    with torch.no_grad():
        for data in valid_dl:
            # evaluate the model on the test set
            yhat = model(data.to(DEVICE))
            #yhat = yhat.reshape(yhat.shape[0], -1)
            actual = data.y.to(DEVICE)
            #actual = actual.reshape(yhat.shape[0], -1)
            # store
            predictions.append(yhat)
            actuals.append(actual)
        yhat.cpu()
        actual.cpu()
        data.cpu()
        
        predictions, actuals = torch.vstack(predictions), torch.vstack(actuals)
        predictions.cpu()
        actuals.cpu()
        #print(predictions.shape, actuals.shape)
        # calculate errors
        
        mse = torch.mean(((predictions * target_stdev_P + target_mean_P) - (actuals * target_stdev_P + target_mean_P))**2)
        
        rmse = torch.sqrt(mse)
        
        nrmse = rmse / target_mean_P

        mae = (abs((predictions * target_stdev_P + target_mean_P) - (actuals * target_stdev_P + target_mean_P))).mean()
        
        nmae = mae / target_mean_P
        
        try:
            mape = (abs(((predictions * target_stdev_P + target_mean_P) - (actuals * target_stdev_P + target_mean_P))/(actuals * target_stdev_P + target_mean_P))).mean()
        except:
            mape = 10e5
        
        mbe = ((predictions * target_stdev_P + target_mean_P) - (actuals * target_stdev_P + target_mean_P)).mean()
        
        skill = 1 - rmse / persistRMSE
        
        r2score = R2Score().to(DEVICE)
        R2 = r2score((predictions * target_stdev_P + target_mean_P), (actuals * target_stdev_P + target_mean_P))
        
        if plot:
            Vactuals = actuals * target_stdev_P + target_mean_P
            Vpredictions = predictions * target_stdev_P + target_mean_P
            df_tmp = pd.DataFrame()
            df_tmp['actuals'] = Vactuals.reshape(-1).cpu()
            df_tmp['predictions'] = Vpredictions.reshape(-1).cpu()
            df_tmp = df_tmp[df_tmp['actuals']>=1]
            
            # ax = plt.figure(figsize=(2.5*5,2.5*5)).add_subplot(projection='3d')
            # ax.plot(df_tmp['actuals'], df_tmp['predictions'], 'o', alpha=0.2,
            #         zs=0, zdir='z', label=None)
            # ax.plot(np.array([min(df_tmp['actuals']), max(df_tmp['actuals'])]),
            #         np.array([min(df_tmp['actuals']), max(df_tmp['actuals'])]), color='purple', linewidth=4,
            #         zs=0, zdir='z', label=None)
            # ax.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
            # ax.axhline(y = 400, color = 'k', linestyle = 'dashed', linewidth = 4)
            # ax.axvline(x = 400, color = 'k', linestyle = ':', linewidth = 3)
            # ax.axvline(x = 400, color = 'k', linestyle = 'dashed', linewidth = 4)
            
            # hist, bins = np.histogram(df_tmp['actuals'], bins=50)
            # xs = (bins[:-1] + bins[1:])/2
            # ax.bar(xs, hist, zs=max(df_tmp['actuals']), zdir='y', alpha=0.9, color='tab:green')
            # hist, bins = np.histogram(df_tmp['predictions'], bins=50)
            # xs = (bins[:-1] + bins[1:])/2
            # ax.bar(xs, hist, zs=min(df_tmp['actuals']), zdir='x', alpha=0.9, color='midnightblue')
            
            # # Make legend, set axes limits and labels
            # #ax.legend()
            # #ax.set_xlim(0, 1)
            # #ax.set_ylim(0, 1)
            # #ax.set_zlim(0, 1)
            # ax.set_xlabel('Measured', size=14)
            # ax.set_ylabel('Forecasted', size=14)
            # ax.set_zlabel('Frequency', size=14)
            
            # #ax.view_init(elev=30., azim=-45)
            # plt.show()
            
            #sns.set_theme(style="whitegrid", color_codes=True, palette="pastel")
            # g = sns.jointplot(x=df_tmp['actuals'], y=df_tmp['predictions'], kind="reg", height=10, ratio=2,
            #                   scatter_kws={'alpha': 0.2},
            #                   marginal_kws={'color': 'tab:green', 'bins': 50, 'kde': True, 'linewidth': 0.5})
            # plt.setp(g.ax_marg_y.patches, color="midnightblue")
            # plt.show()


            g = sns.JointGrid(height=10, 
                              ratio=2,
                              space=0.1,
                              marginal_ticks=True)
            x, y = df_tmp['actuals'], df_tmp['predictions']
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
            plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], Vactuals.reshape(-1).cpu(), label='Measured Discharge', color='tab:green', linewidth=2.5)
            plt.fill_between(df_ML_valid.reset_index(drop=True).iloc[:, 0], Vactuals.reshape(-1).cpu(), color='tab:green', alpha=0.4)
            plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], Vpredictions.reshape(-1).cpu(), linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
            plt.legend(shadow=True, fontsize=14)
            plt.gcf().autofmt_xdate()
            plt.title('ChebConv-GNN Regression - Validation Dataset - ' + str(STUDY_STATION), size=20)
            plt.xlabel('Date', size=14)
            plt.ylabel('Discharge (m3/s)', size=14)
            #plt.ylim(74, 80)
            plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
            plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
            plt.show()
            
            plt.figure(figsize=(5.5*5,2.5*5))
            plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], Vactuals.reshape(-1).cpu(), label='Measured Discharge', color='tab:green')
            plt.fill_between(df_ML_valid.iloc[:, 0], Vactuals.reshape(-1).cpu(), color='tab:green', alpha=0.4)
            plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], Vpredictions.reshape(-1).cpu(), linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
            plt.gcf().autofmt_xdate()
            plt.xlabel('Date', size=14)
            plt.ylabel('Discharge (m3/s)', size=14)
            plt.xlim([pd.to_datetime('2013-06-15', format = '%Y-%m-%d'),
                     pd.to_datetime('2013-07-15', format = '%Y-%m-%d')])
            #plt.ylim(74, 80)
            plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
            plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
            plt.show()
            
            plt.figure(figsize=(5.5*5,2.5*5))
            plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], Vactuals.reshape(-1).cpu(), label='Measured Discharge', color='tab:green')
            plt.fill_between(df_ML_valid.iloc[:, 0], Vactuals.reshape(-1).cpu(), color='tab:green', alpha=0.4)
            plt.plot(df_ML_valid.reset_index(drop=True).iloc[:, 0], Vpredictions.reshape(-1).cpu(), linestyle = 'dashdot', linewidth=2.5, label='Forecasted Discharge', color='midnightblue')
            plt.gcf().autofmt_xdate()
            plt.xlabel('Date', size=14)
            plt.ylabel('Discharge (m3/s)', size=14)
            plt.xlim([pd.to_datetime('2013-06-19', format = '%Y-%m-%d'),
                     pd.to_datetime('2013-06-23', format = '%Y-%m-%d')])
            #plt.ylim(74, 80)
            plt.axhline(y = 400, color = 'k', linestyle = ':', linewidth = 3)
            plt.axhline(y = 500, color = 'k', linestyle = 'dashed', linewidth = 4)
            plt.show()
            
            del Vpredictions, Vactuals
        
        if evalEPA: # performs the rolling mean (8hs-EPA-Ozone) to count how many times the model can catch the the occurence of levels above the permitted

            Vactuals = actuals.cpu() * target_stdev_P + target_mean_P
            Vactuals.cpu()
            Vpredictions = predictions.cpu() * target_stdev_P + target_mean_P
            Vpredictions.cpu()
            
            limitSup = torch.round(torch.quantile(Vactuals, 0.80), decimals=1)
            print(limitSup)

            df_EPA_hour = pd.DataFrame()
            df_EPA_hour['DTG'] = df_ML_valid.reset_index(drop=True).iloc[:, 0]
            df_EPA_hour['Meas'] = Vactuals
            df_EPA_hour['Pred'] = Vpredictions
            # 8-hour rolling average
            df_EPA_hour['Meas_roll_mean'] = df_EPA_hour['Meas'].rolling(1).mean()   # for EPA, .rolling(8)
            df_EPA_hour['Pred_roll_mean'] = df_EPA_hour['Pred'].rolling(1).mean()
            
            print(df_EPA_hour)
            
            # daily maxima
            df_EPA_day = pd.DataFrame()
            df_EPA_day['Date'] = pd.to_datetime(df_ML_valid.reset_index(drop=True).iloc[:, 0]).dt.date
            df_EPA_day['max_meas']=df_EPA_hour.groupby(df_EPA_hour.DTG.dt.date)['Meas_roll_mean'].transform(max)
            df_EPA_day['max_pred']=df_EPA_hour.groupby(df_EPA_hour.DTG.dt.date)['Pred_roll_mean'].transform(max)
            df_EPA_day.drop_duplicates(inplace=True)
            df_EPA_day.sort_values(by='max_pred', ascending=False, inplace=True)
            df_EPA_day.reset_index(drop=True, inplace=True)
            
            print(df_EPA_day.sort_values(by='max_meas', ascending=False)['max_meas'])
            print(df_EPA_day.sort_values(by='max_meas', ascending=False)['max_pred'])
            
            num_bars = 50
            X = np.arange(num_bars)#(len(df_EPA_day))
            width = 0.25
            fig = plt.figure(figsize=(7*2.5,3*2.5))
            ax = fig.add_axes([0,0,1,1])
            ax.bar(X - width/2, df_EPA_day.sort_values(by='max_meas', ascending=False)['max_meas'][:num_bars], width, label='Measured', color='tab:green')
            ax.bar(X + width/2, df_EPA_day.sort_values(by='max_meas', ascending=False)['max_pred'][:num_bars], width, label='Forecasted', color='midnightblue')
            ax.axhline(y = limitSup, color = 'r', linestyle = ':', linewidth = 4)
            ax.set_ylabel('Level (m)', fontsize=20)
            ax.set_xlabel('highest daily maxima (1-hour average)', fontsize=20)

            #plt.ylim(0,100)
            ax.tick_params(labelsize=16)
            ax.set_xticks(X)
            ax.set_xticklabels(range(1,num_bars+1))
            ax.legend(fontsize=20)
            
            del Vactuals, Vpredictions, limitSup, df_EPA_hour, df_EPA_day, num_bars
        
    del yhat, actual, data, predictions, actuals
    
    model.train()
    
    return mse, rmse, nrmse, mae, nmae, mape, mbe, skill, R2


####################################################################################################################
# Defines errors
####################################################################################################################

def CustomLoss_MAE_RMSE(actual, predicted, wRMSE=1.0, DEVICE='cpu'):
    
    #print(actual.shape, predicted.shape)
    actual = actual.reshape(actual.shape[0], -1)
    predicted = predicted.reshape(predicted.shape[0], -1)
    #print(actual.shape, predicted.shape)
    
    # calculate mean squared error combined with mean absolute error 
    loss = (1-wRMSE) * torch.mean(abs(predicted - actual)) # L1 error parcel
    loss = loss + wRMSE * torch.sqrt(torch.mean((predicted - actual)**2)) # L2 error parcel
       
    return loss


def CustomLoss_Ln(actual, predicted, order=2, DEVICE='cpu'):
    
    #print(actual.shape, predicted.shape)
    actual = actual.reshape(actual.shape[0], -1)
    predicted = predicted.reshape(predicted.shape[0], -1)
    #print(actual.shape, predicted.shape)
    
    # calculate l-n error 
    loss = torch.mean(abs(predicted - actual)**order) # Ln error
       
    return loss


####################################################################################################################
# Defines the training process
####################################################################################################################

def train_model(train_dl, valid_dl, model, DEVICE='cpu', epochs=11, persistRMSE=1.0):
    # define the optimization
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = CustomLoss_MAE_RMSE
    criterion = CustomLoss_Ln
    
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    #from torch.optim import SGD
    #optimizer = SGD(model.parameters(), lr=1.0, momentum=0.9)

    #from torch.optim import RMSprop
    #optimizer = RMSprop(model.parameters(), lr=0.01)
    
    #from torch.optim import AdamW
    #optimizer = AdamW(model.parameters(), lr=0.01)
    
    #from torch.optim import Adadelta
    #optimizer = Adadelta(model.parameters(), lr=1.0)
    
    #from torch.optim import LBFGS
    #optimizer = LBFGS(model.parameters(), lr=0.08)
    
    #from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=13, eps=1e-10)
    
    valid_loss_set = []
    train_loss_set = []
    best_error = 1e7
    early_stop_count = 0
    early_stop_patience = 50
    
    # enumerate epochs
    for epoch in range(epochs):
        print('\nEPOCH -', epoch, '   -   LR -', "{:.2E}".format(optimizer.param_groups[0]['lr']))
        # enumerate mini batches 
        loss_ave = []
        for data in train_dl:
            # clear the gradients
            optimizer.zero_grad()
            # send the minibatch to GPU
            data = data.to(DEVICE)
            # compute the model output
            yhat = model(data)
            # calculate loss
            #loss = criterion(data.y * target_stdev_P + target_mean_P, yhat * target_stdev_P + target_mean_P, wRMSE=1.0, DEVICE=DEVICE) # if criterion is CustomLoss_MAE_RMSE
            loss = criterion(data.y * target_stdev_P + target_mean_P, yhat * target_stdev_P + target_mean_P, order=2, DEVICE=DEVICE) # if criterion is CustomLoss_Ln
            loss_ave.append(loss.cpu().detach().numpy())
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        #print(data.y, 'data.y')
        #print(data.y.shape, 'data.y')
        #print(yhat, 'yhat')
        #print(yhat.shape, 'yhat')
        data.cpu(); del data
        
        train_loss_set.append(np.mean(loss_ave))
        print('Training loss: %.4f' % np.mean(loss_ave))
        
        valid_loss = evaluate_model(valid_dl, model, DEVICE=DEVICE,
                                    persistRMSE=valid_persist_errors[0])
        print('Validation loss: %.4f' % valid_loss[1])
        valid_loss_set.append(valid_loss[1])
        
        # early stopping check
        print('Early stopping: ', early_stop_count, ' / ', early_stop_patience)
        if valid_loss[1] < (0.999 * best_error): # improved error more than 1% - no early stopping
            best_error = valid_loss[1] # update best result
            early_stop_count = 0 # reset early-stop counting
        else: # did not improve error
            early_stop_count += 1
        
        if early_stop_count == early_stop_patience:
            print('\n#### WARNING: Reached the patience limit (', early_stop_patience, ' steps), stopping in epoch ', epoch, ' ... ####')
            break # get out of the loop
        
        scheduler.step(loss)
    loss.cpu(); del loss
    
    # plot train and test evolution
    
    #print(train_loss_set, test_loss_set)
    plt.figure(figsize=(20,10))
    plt.title("Train-Validation Accuracy", fontsize=25)
    plt.plot(torch.as_tensor(train_loss_set), label='train', color='tab:orange')
    plt.plot(torch.as_tensor(valid_loss_set), label='validation', color='tab:blue')
    plt.xlabel('num_epochs', fontsize=20)
    plt.ylabel('MSE', fontsize=20)
    #plt.ylim(0.0,0.2)
    plt.legend(loc='best', fontsize=20)
    plt.show()
    
    del train_loss_set, valid_loss_set


####################################################################################################################
# Train the model
####################################################################################################################
TOC = time.perf_counter() # start clock

# clean up GPU memory
cleans_GPU()

train_model(train_loader, valid_loader, model, DEVICE=DEVICE, epochs=EPOCHS, persistRMSE=1)#valid_persist_errors_V[0])

# clean up GPU memory
cleans_GPU()

TIC = time.perf_counter() # end clock
print(f"\nTrained the model in {TIC - TOC:0.4f} seconds \n")


####################################################################################################################
# Evaluate the model
####################################################################################################################
valid_ChebConv_errors = evaluate_model(valid_loader, model, DEVICE=DEVICE,
                                   persistRMSE=valid_persist_errors[0], plot=plotChebConv, evalEPA=False)

valid_ChebConv_errors = list(valid_ChebConv_errors)
for i in range(len(valid_ChebConv_errors)):
    valid_ChebConv_errors[i] = valid_ChebConv_errors[i].item()

print('\nFinal results for the validation set:')

print(f"\nChebConv Validation RMSE = {valid_ChebConv_errors[1]:.5f}")
print(f"ChebConv Validation nRMSE = {valid_ChebConv_errors[2]:.5f}")
print(f"ChebConv Validation MAE = {valid_ChebConv_errors[3]:.5f}")
print(f"ChebConv Validation nMAE = {valid_ChebConv_errors[4]:.5f}")
print(f"ChebConv Validation MAPE = {valid_ChebConv_errors[5]:.5f}")
print(f"ChebConv Validation MBE = {valid_ChebConv_errors[6]:.5f}")
print(f"ChebConv Validation Skill = {valid_ChebConv_errors[7]:.5f}")
print(f"ChebConv Validation R2 = {valid_ChebConv_errors[8]:.5f}")
print()


####################################################################################################################
# Plot final results
####################################################################################################################

# for inst in range(0, int(0.8 * totalTime / timeResolution), int(0.2 * totalTime / timeResolution)):
#       print('\nInstant t = ', inst, '\n')
#       mylib.plot_forecasted_seq(inst, spatialResolution, timeResolution, model,
#                                 Tsample,
#                                 TsampleMean, TsampleStd,
#                                 plotOutputTimeSteps=finalPlotOutputTimeSteps,
#                                 stepSize=finalPlotStepSize,
#                                 modelType=modelType)

print('\nMODEL STRUCTURE:\n')
print(model)


###############################################################################
# THE END
###############################################################################
