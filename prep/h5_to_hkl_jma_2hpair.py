#
# convert jma rada data in h5 format into hkl data
# for prednet
#
import os
import glob

import numpy as np
import hickle as hkl
import h5py
import pandas as pd

img_size = 128

def save_as_hickle(dname,data_dir,csv_file,case):
    print('save_as_hickle',dname,data_dir,csv_file,case)
    
    df_fnames = pd.read_csv(csv_file)
    #print(list)
    nt = 24 # time steps per data (1h)
    X_sources = np.zeros(len(df_fnames)*nt)
    X_all = np.zeros((len(df_fnames)*nt,img_size,img_size,1),dtype=np.float16)
    print('data size :%d Gbytes' % (X_all.size * X_all.itemsize/1.0e9))

    for i in range(len(df_fnames)):
        # read Past data
        h5_name_X = os.path.join(dname, data_dir, df_fnames.ix[i, 'fname'])
        print('reading:',i,h5_name_X)
        h5file = h5py.File(h5_name_X,'r')
        X = h5file['R'][()]
        X = np.maximum(X,0) # replace negative value with 0
        X = X/201.0*255.0 # scale range to [0-255]
        X = X[:,:,:,None] # add "channel" dimension as 1 (channel-last format)
        h5file.close()
        # read Future data
        h5_name_Y = os.path.join(dname, data_dir, df_fnames.ix[i, 'fnext'])
        print('reading:',i,h5_name_Y)
        h5file = h5py.File(h5_name_Y,'r')
        Y = h5file['R'][()]
        Y = np.maximum(Y,0) # replace negative value with 0
        Y = Y/201.0*255.0 # scale range to [0-255]
        Y = Y[:,:,:,None] # add "channel" dimension as 1 (channel-last format)
        # save
        i1 = i*nt
        i2 = (i+1)*nt
        XY = np.concatenate([X,Y],axis=0)
        X_all[i1:i2,:,:,:] = XY
        X_sources[i1:i2] = i

    hkl.dump(X_all, dname+case+'_data.hkl', mode='w')
    hkl.dump(X_sources, dname+case+'_sources.hkl', mode='w')

dname = '../data/jma/'
data_dir = 'data_kanto_resize'
# 2015 and 2016 data for training
tr_csv = '../data/jma/train_simple_JMARadar.csv'
save_as_hickle(dname,data_dir,tr_csv,'jma_2hr_128_train_2015-2016')
#tr_csv = '../data/jma/train_kanto_flatsampled_JMARadar.csv'
#save_as_hickle(dname,data_dir,tr_csv,'jma_2hr_train_flatsampled')
# 2017 data to test
va_csv = '../data/jma/valid_simple_JMARadar.csv'
save_as_hickle(dname,data_dir,va_csv,'jma_2hr_128_test_2017')
