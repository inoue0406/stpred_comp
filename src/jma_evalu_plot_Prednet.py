'''
Plot trained PredNet on JMA radar dataset
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
from plot_comp_jma import *

from criteria_precip import *

batch_size = 1
#nt = 12
#nt_1stp = 6 #time dimension for one-step prediction
nt = 24
nt_1stp = 12 #time dimension for one-step prediction

# scale factor for converting [0-1] range data to [0-201.0] mm/h
scale_factor = 201.0

#case = 'case_190825_jma_Prednet_nt80'
#case = 'case_190901_jma_Prednet_128_nt80'
case = 'case_190903_jma_Prednet-ft_128_nt80'

#weights_file = os.path.join(WEIGHTS_DIR, case, 'prednet_jma_weights.hdf5')
weights_file = os.path.join(WEIGHTS_DIR, case, 'prednet_jma_weights-finetuned.hdf5')
json_file = os.path.join(WEIGHTS_DIR, case, 'prednet_jma_model-finetuned.json')

#test_file = os.path.join(DATA_DIR, 'jma_test_2017_data.hkl')
#test_sources = os.path.join(DATA_DIR, 'jma_test_2017_sources.hkl')
test_file = os.path.join(DATA_DIR, 'jma_2hr_128_test_2017_data.hkl')
test_sources = os.path.join(DATA_DIR, 'jma_2hr_128_test_2017_sources.hkl')

#plot cases list
plot_cases = os.path.join(DATA_DIR, 'sampled_forplot_3day_JMARadar.csv')
df_sampled = pd.read_csv(plot_cases)

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

# Prep data
test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format, batch_size=batch_size)

# list of indices to plot
idx_plot = df_sampled['index'].values

# First load all the data, then select by index
X_test = test_generator.create_all()
X_test = X_test[idx_plot,:,:,:,:]
for i in range(len(idx_plot)):
    print('index:',idx_plot[i],', :max value',np.max(X_test[i,:,:,:,:]))

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
        #print('prediction with steps from ',n1,' to ',n2,' \n')
        X_t1 = X_tmp[:,n1:n2,:,:,:]
        X_h1 = test_model.predict(X_t1, len(idx_plot))
        # prediction for 1step
        X_hat[:,n2,:,:,:] = X_h1[:,ntpred-1,:,:,:]
        # use the prediction as the next input
        X_tmp[:,n2,:,:,:] = X_h1[:,ntpred-1,:,:,:]
    return X_hat

# prep path
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, case, 'prediction_plots/')
if not os.path.exists(plot_save_dir):
    os.mkdir(plot_save_dir)

# loop through data loader steps
X_hat = predict_multistep(nt,nt_1stp,X_test,test_model)

# convert to 0-201(mm/h) range
X_test = X_test * scale_factor
X_hat = X_hat * scale_factor
    
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

npics = len(idx_plot)
#plot_comp_prediction(X_test,X_hat,nt,nt_1stp,npics,plot_save_dir,case,mode='png_whole')
plot_comp_prediction(X_test,X_hat,nt,nt_1stp,npics,plot_save_dir,case,mode='png_ind')

