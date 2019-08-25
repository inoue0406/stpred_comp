#
# convert jma rada data in h5 format into hkl data
# for prednet
#
import os
import glob

import numpy as np
import hickle as hkl
import h5py

def save_as_hickle(dname,pattern,case):
    print('save_as_hickle',dname,pattern,case)
    #pattern = '../data/jma/data_kanto_tmp/*h5'
    #flist = os.listdir(dirname)
    flist = glob.glob(pattern)
    flist.sort()

    #print(list)
    nt = 12 # time steps per data (1h)
    X_sources = np.zeros(len(flist)*nt)
    X_all = np.zeros((len(flist)*nt,200,200,1),dtype=np.float16)
    print('data size :%d Gbytes' % (X_all.size * X_all.itemsize/1.0e9))

    for i,fname in enumerate(flist):
        print('reading:',i,fname)
        h5file = h5py.File(fname,'r')
        X = h5file['R'][()]
        X = np.maximum(X,0) # replace negative value with 0
        X = X/201.0*255.0 # scale range to [0-255]
        X = X[:,:,:,None] # add "channel" dimension as 1 (channel-last format)
        h5file.close()
        i1 = i*nt
        i2 = (i+1)*nt
        X_all[i1:i2,:,:,:] = X
        X_sources[i1:i2] = i

    hkl.dump(X_all, dname+case+'_data.hkl', mode='w')
    hkl.dump(X_sources, dname+case+'_sources.hkl', mode='w')

dname = '../data/jma/'
# 2015 and 2016 data for training 
save_as_hickle(dname,'../data/jma/data_kanto/2p-jmaradar5_201[56]*h5','jma_train_2015-2016')
#save_as_hickle(dname,'../data/jma/data_kanto/2p-jmaradar5_2015*h5','jma_train_2015')
# 2017 data to test
save_as_hickle(dname,'../data/jma/data_kanto/2p-jmaradar5_2017*h5','jma_test_2017')
