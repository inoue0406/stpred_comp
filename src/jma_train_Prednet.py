'''
Train PredNet on JMA radar data
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

# A config to allow GPU memory check by nvidia-smi
if 'tensorflow' == K.backend():
    import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from prednet import PredNet
from data_utils import SequenceGenerator
from settings_jma import *

#case = 'case_190825_jma_Prednet_nt80'
case = 'case_190901_jma_Prednet_128_nt80'

save_model = True  # if weights will be saved
result_dir = os.path.join(WEIGHTS_DIR, case)
weights_file = os.path.join(result_dir, 'prednet_jma_weights.hdf5')
json_file = os.path.join(result_dir, 'prednet_jma_model.json')
print('result dir path:',result_dir,'\n')

# Data files
#train_file = os.path.join(DATA_DIR, 'jma_train_2015-2016_data.hkl')
#train_sources = os.path.join(DATA_DIR, 'jma_train_2015-2016_sources.hkl')
#val_file = os.path.join(DATA_DIR, 'jma_test_2017_data.hkl')
#val_sources = os.path.join(DATA_DIR, 'jma_test_2017_sources.hkl')
train_file = os.path.join(DATA_DIR, 'jma_2hr_128_train_2015-2016_data.hkl')
train_sources = os.path.join(DATA_DIR, 'jma_2hr_128_train_2015-2016_sources.hkl')
val_file = os.path.join(DATA_DIR, 'jma_2hr_128_test_2017_data.hkl')
val_sources = os.path.join(DATA_DIR, 'jma_2hr_128_test_2017_sources.hkl')

# Training parameters
nb_epoch = 80
batch_size = 5
samples_per_epoch = 3000
N_seq_val = 3000  # number of sequences to use for validation

# Model parameters
n_channels, im_height, im_width = (1, 128, 128)
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
#nt = 12  # number of timesteps used for sequences in training
nt = 24  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

# print model summary 
model.summary()

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                validation_data=val_generator, validation_steps=N_seq_val / batch_size)

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
