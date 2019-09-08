'''
Plot trained PredNet on JMA radar dataset
multi-step forecast
'''

import os
import sys
import numpy as np
import matplotlib

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)

from data_utils import SequenceGenerator
from settings_jma import *

batch_size = 1
nt = 24
nt_1stp = 12 #time dimension for one-step prediction

# scale factor for converting [0-1] range data to [0-201.0] mm/h
scale_factor = 201.0

test_file = os.path.join(DATA_DIR, 'jma_2hr_128_train_2015-2016_data.hkl')
test_sources = os.path.join(DATA_DIR, 'jma_2hr_128_train_2015-2016_sources.hkl')

# Prep data
test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format='channels_last', batch_size=batch_size)

# First load all the data, then select by index
X_test = test_generator.create_all()
X_test = X_test

def print_range_by_value(X_test,value,fwrite):
    
    f = open(RESULTS_SAVE_DIR + fwrite, 'w')

    for i in range(X_test.shape[0]):
        print('index:',i,', :max value',np.max(X_test[i,:,:,:,:]))
        Xtmp = np.max(X_test[i,:,:,:,:],axis=(1,2,3))
        Xtmp = Xtmp * scale_factor
        #value = 10.0
        vrange = 0.5
        # get id within value +- vrange
        range_id = np.where((Xtmp > (value-vrange)) * (Xtmp < (value+vrange)))[0]
        range_id = range_id[(range_id >= 5) * (range_id <= 18)]
        if len(range_id) > 0:
            id_slct = range_id[0]
            # select +-5 range
            xprint = Xtmp[(id_slct-5):(id_slct)+6]
            print("selected ",i,xprint)
            f.write("%d, %d, " % (i,id_slct))
            np.savetxt(f, xprint, delimiter=',',newline=',')
            f.write("\n")
            
    f.close()
    return

print_range_by_value(X_test,5.0,'stat_rain_jma_5mm.txt')
print_range_by_value(X_test,10.0,'stat_rain_jma_10mm.txt')
print_range_by_value(X_test,20.0,'stat_rain_jma_20mm.txt')
print_range_by_value(X_test,30.0,'stat_rain_jma_30mm.txt')
print_range_by_value(X_test,50.0,'stat_rain_jma_50mm.txt')


