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

# loop through 1-step prediction to get multi-step preictions
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

if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(os.path.join(RESULTS_SAVE_DIR, case, 'prediction_scores.txt'), 'w')
# loop through evary time for estimation/prediction
f.write("time, MSE, MAE, SSIM\n")
for i in range(X_test.shape[1]):
    mse = np.mean( (X_test[:,i,:,:,:] - X_hat[:,i,:,:,:])**2 )  # MSE for time i
    mae = np.mean( np.abs(X_test[:,i,:,:,:] - X_hat[:,i,:,:,:]) )  # MAE for time i
    ssim = 0.0
    for k in range(X_test.shape[0]):
        ssim += evaluu.compare_ssim(X_test[k,i,:,:,0],X_hat[k,i,:,:,0])
    ssim = ssim / X_test.shape[0]
    #ssim= evaluu.compare_ssim(X_test[:,i,:,:,:],X_hat[:,i,:,:,:],win_size=3,multichannel=True)
    f.write("%f,%f,%f,%f\n" % (i,mse,mae,ssim))
f.close()

# Plot some predictions

plot_save_dir = os.path.join(RESULTS_SAVE_DIR, case, 'prediction_plots/')
npics = 10
plot_comp_prediction(X_test,X_hat,nt,nt_1stp,npics,plot_save_dir,case,mode='png_whole')
plot_comp_prediction(X_test,X_hat,nt,nt_1stp,npics,plot_save_dir,case,mode='png_ind')

