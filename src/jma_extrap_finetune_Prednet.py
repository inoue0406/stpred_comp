'''
Fine-tuning for jma data
'''

import os
import numpy as np
np.random.seed(123)

from keras import backend as K
from keras.models import Model,model_from_json
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

# Define loss as MAE of frame predictions after t=0
# It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when extrapolating.
def extrap_loss(y_true, y_hat):
    y_true = y_true[:, 1:]
    y_hat = y_hat[:, 1:]
    return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)  # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

#case = 'case_190825_jma_Prednet_nt80'
case = 'case_190901_jma_Prednet_128_nt80'

nt = 24
extrap_start_time = 12  # starting at this time step, the prediction from the previous time step will be treated as the actual input
orig_weights_file = os.path.join(WEIGHTS_DIR, case, 'prednet_jma_weights.hdf5')# original t+1 weights
orig_json_file = os.path.join(WEIGHTS_DIR, case, 'prednet_jma_model.json')

save_model = True  # if weights will be saved
result_dir = os.path.join(WEIGHTS_DIR, case)
extrap_weights_file = os.path.join(result_dir, 'prednet_jma_weights-finetuned.hdf5')
extrap_json_file = os.path.join(result_dir, 'prednet_jma_model-finetuned.json')
print('result dir path:',result_dir,'\n')

# Data files
train_file = os.path.join(DATA_DIR, 'jma_2hr_128_train_2015-2016_data.hkl')
train_sources = os.path.join(DATA_DIR, 'jma_2hr_128_train_2015-2016_sources.hkl')
val_file = os.path.join(DATA_DIR, 'jma_2hr_128_test_2017_data.hkl')
val_sources = os.path.join(DATA_DIR, 'jma_2hr_128_test_2017_sources.hkl')

# Training parameters
nb_epoch = 80
batch_size = 5
samples_per_epoch = 3000
N_seq_val = 3000  # number of sequences to use for validation

# Load t+1 model
f = open(orig_json_file, 'r')
json_string = f.read()
f.close()
orig_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
orig_model.load_weights(orig_weights_file)


layer_config = orig_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
layer_config['extrap_start_time'] = extrap_start_time
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
prednet = PredNet(weights=orig_model.layers[1].get_weights(), **layer_config)

input_shape = list(orig_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt

inputs = Input(input_shape)
predictions = prednet(inputs)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=extrap_loss, optimizer='adam')

train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True, output_mode='prediction')
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val, output_mode='prediction')

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=extrap_weights_file, monitor='val_loss', save_best_only=True))

# Train the model
history = model.fit_generator(train_generator, samples_per_epoch / batch_size, nb_epoch, callbacks=callbacks,
                              validation_data=val_generator, validation_steps=N_seq_val / batch_size)

# Save trained model
if save_model:
    json_string = model.to_json()
    with open(extrap_json_file, "w") as f:
        f.write(json_string)
