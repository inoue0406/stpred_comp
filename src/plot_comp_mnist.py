import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_comp_prediction(X_test,X_hat,nt,nt_1stp,npics,
                         pic_path,case,mode='png_whole'):
    """
    Compare the plots of predicted vs ground truth

    Args:
    X_test : Ground truth tensor to be plotted 
             Dimension should be (sample,time,height,width,channels)
    X_hat : Predicted tensor which has the same shape as X_test
    nt : time dimension
    nt_1stp : time dim used in 1step prediction
    npics : number of samples that you want to plot
    pic_path : output dir for png files
    case  : case name

    Returns: None

    """
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # number of predicted steps
    ntpred = nt-nt_1stp
    
    #for i in range(X_test.shape[0]):
    for i in range(npics):
        print('Plotting for sample:',i)
        # plot
        cm = 'Greys'
        if mode == 'png_whole': # output as stationary image
            fig, ax = plt.subplots(figsize=(20, 6))
            fig.suptitle("Prediction: "+case, fontsize=20)
            for n in range(nt_1stp,nt):
                id = n
                pos = n+1-nt_1stp
                dtstr = str((id+1))
                # target
                plt.subplot(2,ntpred,pos)
                im = plt.imshow(X_test[i,n,:,:,:].squeeze(),
                                vmin=0,vmax=1,cmap=cm,origin='upper')
                plt.title("true:"+dtstr+"steps")
                plt.grid()
                # predicted
                plt.subplot(2,ntpred,pos+ntpred)
                im = plt.imshow(X_hat[i,n,:,:,:].squeeze(),
                                vmin=0,vmax=1,cmap=cm,origin='upper')
                plt.title("pred:"+dtstr+"steps")
                plt.grid()
            fig.subplots_adjust(right=0.95)
            cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            # save as png
            plt.savefig(pic_path+'comp_pred_sample'+str(i)+'.png')
            plt.close()
        if mode == 'png_ind': # output as invividual image
            for n in range(nt_1stp,nt):
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.suptitle("Prediction: "+case, fontsize=20)
                #        
                id = n
                pos = n+1-nt_1stp
                dtstr = str((id+1))
                # target
                plt.subplot(1,2,1)
                im = plt.imshow(X_test[i,n,:,:,:].squeeze(),
                                vmin=0,vmax=1,cmap=cm,origin='upper')
                plt.title("true:"+dtstr+"steps")
                plt.grid()
                # predicted
                plt.subplot(1,2,2)
                im = plt.imshow(X_hat[i,n,:,:,:].squeeze(),
                                vmin=0,vmax=1,cmap=cm,origin='upper')
                plt.title("pred:"+dtstr+"steps")
                plt.grid()
                # color bar
                fig.subplots_adjust(right=0.93,top=0.85)
                cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                n_str = '_dt%02d' % n
                plt.savefig(pic_path+'comp_pred_sample'+str(i)+n_str+'.png')
                plt.close()
        
