#
# convert jma rada data in h5 format into hkl data
# for prednet
# 
import numpy as np
import hickle as hkl

hkl.dump(in_video, dname+fname+'_data.hkl', mode='w')
    
save_as_hickle(0,6000,dname,'mnist_tain_6000')
save_as_hickle(6000,8000,dname,'mnist_valid_2000')
save_as_hickle(8000,10000,dname,'mnist_test_2000')


