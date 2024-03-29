from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
# author Nicolas Riss (nicolas.riss22@gmail.com)

##############
# parameters #
##############
folderName = "data_test_beads" #test: 'data_test'
filename = "patchTest.npz" #test: "patchTest.npz"
modelName = "modelTest" #test: "modelTest"
baseDir = "models" #test: "models" name of the directory containing the model
stepPerEpoch = 100 #test: '100'  can be increased considerably for a well-train model (ex: 400)
validationSplit = 0.1 #test: 0.1 Percentage of patches conserved for validation

#################
# TRAINING DATA #
#################
#10% of validation data are used there.
(X_train,Y_train), (X_val,Y_val), axes = load_training_data(folderName + '/' + filename, validation_split=validationSplit, verbose=True)
#(X_train, Y_train), (X_val,Y_val), axes = load_training_data('data/synthetic_disks/data.npz', validation_split=0.1, verbose=True)

print("axes : ", axes)

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X_train.shape[c], Y_train.shape[c]


plt.figure(figsize=(12,5))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)');

#################
# Configuration #
#################

# Config object contains: parameters of the underlying neural network, learning rate, number of parameter updates per epoch, loss function, and whether the model is probabilistic or not.

config = Config(axes, n_channel_in, n_channel_out, probabilistic=True, train_steps_per_epoch=stepPerEpoch)
print(config)
vars(config)

############
# TRAINING #
############
#Possibility to monitor the progress using TensorBoat (see https://www.tensorflow.org/guide/summaries_and_tensorboard)

# model instanciation
#model = CARE(config=None, name='my_model', basedir='models') # used to load a model
model = CARE(config, modelName, basedir=baseDir) # used to train a new model

# training model
history = model.train(X_train,Y_train, validation_data=(X_val,Y_val))


#########################################
# Show 5 examples of validation patches #
#########################################
plt.figure(figsize=(12,10))
_P = model.keras_model.predict(X_val[:5])
_P_mean  = _P[...,:(_P.shape[-1]//2)]
_P_scale = _P[...,(_P.shape[-1]//2):]
plot_some(X_val[:5],Y_val[:5],_P_mean,_P_scale,pmax=99.5)
plt.suptitle('5 example validation patches\n'
             'first row: input (source),  '
             'second row: target (ground truth),  '
             'third row: predicted Laplace mean,  '
             'forth row: predicted Laplace scale');
plt.show()



#exporting model with Fiji Plugin https://github.com/CSBDeep/CSBDeep_website/wiki/Your-Model-in-Fiji
model.export_TF()
