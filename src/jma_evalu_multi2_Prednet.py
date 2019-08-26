'''
Evaluate trained PredNet on Moving MNIST sequences.
multi-step forecast
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from skimage import measure as evaluu

from prednet import PredNet
from data_utils import SequenceGenerator
from settings_jma import *
from plot_comp_mnist import *

from criteria_precip import *

batch_size = 5
nt = 12
nt_1stp = 6 #time dimension for one-step prediction

case = 'case_190820_jma_Prednet_nt20'

weights_file = os.path.join(WEIGHTS_DIR, case, 'prednet_jma_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, case, 'prednet_jma_model.json')
test_file = os.path.join(DATA_DIR, 'jma_test_2017_data.hkl')
test_sources = os.path.join(DATA_DIR, 'jma_test_2017_sources.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt_1stp
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format, batch_size=batch_size)

# first, load the whole data
#X_test = test_generator.create_all()
# if you want to test with smaller dataset
X_test = X_test[0:100,:,:,:,:]

#import pdb;pdb.set_trace()

def predict_multistep(nt,nt_1stp,X_test,test_model):
    '''
    Multi-step prediction using PredNet model

    '''
    X_hat = np.zeros(X_test.shape,dtype=np.float32)
    X_tmp = X_test.copy()
    ntpred = nt-nt_1stp
    for n in range(ntpred):
        n1 = n
        n2 = nt_1stp + n
        print('prediction with steps from ',n1,' to ',n2,' \n')
        X_t1 = X_tmp[:,n1:n2,:,:,:]
        X_h1 = test_model.predict(X_t1, batch_size)
        # prediction for 1step
        X_hat[:,n2,:,:,:] = X_h1[:,ntpred-1,:,:,:]
        # use the prediction as the next input
        X_tmp[:,n2,:,:,:] = X_h1[:,ntpred-1,:,:,:]
    return X_hat

# prep path
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)

# initialize
SumSE_all = np.empty((0,nt),float)
hit_all = np.empty((0,nt),float)
miss_all = np.empty((0,nt),float)
falarm_all = np.empty((0,nt),float)
m_xy_all = np.empty((0,nt),float)
m_xx_all = np.empty((0,nt),float)
m_yy_all = np.empty((0,nt),float)
MaxSE_all = np.empty((0,nt),float)
FSS_t_all = np.empty((0,nt),float)

threshold = 0.5
# loop through  steps
for i in range(len(test_generator)):
    # use only "x"
    X_test,_ = test_generator[i]
    X_hat = predict_multistep(nt,nt_1stp,X_test,test_model)
    
    if data_format == 'channels_first':
        X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
        X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

    # apply various evaluation metric
    SumSE,hit,miss,falarm,m_xy,m_xx,m_yy,MaxSE = StatRainfall(X_test,X_hat,
                                                              th=threshold)
    FSS_t = FSS_for_tensor(Xtrue,Xmodel,th=threshold,win=10)
        
    SumSE_all = np.append(SumSE_all,SumSE,axis=0)
    hit_all = np.append(hit_all,hit,axis=0)
    miss_all = np.append(miss_all,miss,axis=0)
    falarm_all = np.append(falarm_all,falarm,axis=0)
    m_xy_all = np.append(m_xy_all,m_xy,axis=0)
    m_xx_all = np.append(m_xx_all,m_xx,axis=0)
    m_yy_all = np.append(m_yy_all,m_yy,axis=0)
    MaxSE_all = np.append(MaxSE_all,MaxSE,axis=0)
    FSS_t_all = np.append(FSS_t_all,FSS_t,axis=0)
    
# logging for epoch-averaged loss
RMSE,CSI,FAR,POD,Cor,MaxMSE,FSS_mean = MetricRainfall(SumSE_all,hit_all,miss_all,falarm_all,
                                                      m_xy_all,m_xx_all,m_yy_all,
                                                      MaxSE_all,FSS_t_all,axis=(0))
# save evaluated metric as csv file
tpred = (np.arange(opt.tdim_use)+1.0)*5.0 # in minutes
df = pd.DataFrame({'tpred_min':tpred,
                   'RMSE':RMSE,
                   'CSI':CSI,
                   'FAR':FAR,
                   'POD':POD,
                   'Cor':Cor,
                   'MaxMSE': MaxMSE,
                   'FSS_mean': FSS_mean})
df.to_csv(os.path.join(RESULTS_SAVE_DIR, case,
                       'test_evaluation_predtime_%.2f.csv' % threshold))

#plot_save_dir = os.path.join(RESULTS_SAVE_DIR, case, 'prediction_plots/')
#npics = 10
#plot_comp_prediction(X_test,X_hat,nt,nt_1stp,npics,plot_save_dir,case,mode='png_whole')
#plot_comp_prediction(X_test,X_hat,nt,nt_1stp,npics,plot_save_dir,case,mode='png_ind')

