##############
# TEST MODEL #
##############
'''
    Python script used to test the model after computing
'''

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

(X_train,Y_train), (X_val,Y_val), axes = load_training_data('test.npz', validation_split=0.1, verbose=True)

model = CARE(config=None, name='my_model', basedir='models')
# training model
#history = model.train(X_train,Y_train, validation_data=(X_val,Y_val))

plt.figure(figsize=(12,10))
_P = model.keras_model.predict(X_val[3:8])
_P_mean  = _P[...,:(_P.shape[-1]//2)]
_P_scale = _P[...,(_P.shape[-1]//2):]
plot_some(X_val[3:8],Y_val[3:8],_P_mean,_P_scale,pmax=99.5)
plt.suptitle('5 example validation patches\n'
             'first row: input (source),  '
             'second row: target (ground truth),  '
             'third row: predicted Laplace mean,  '
             'forth row: predicted Laplace scale');
plt.show()


y = imread("data_ome_tif/net/1_MMStack_Pos0.ome.tif")
x = imread("data_ome_tif/flou/1_MMStack_Pos0.ome.tif")

#x = imread("1_20mw_1_MMStack_Pos0.ome.tif")
restored = model.predict(x[20], "YX", normalizer=None) #axes?
# print(type(test))
# print(test.ndim, test.shape)

from csbdeep.io import save_tiff_imagej_compatible
save_tiff_imagej_compatible('testresult.tif', restored, "YX")


plt.figure(figsize=(16,10))
plot_some(np.stack([x[20],restored]),
          title_list=[['low (maximum projection)','CARE (maximum projection)']],
          pmin=2,pmax=99.8);

plt.show()

restored_prob = model.predict_probabilistic(x[20], "YX", normalizer=None) #axes?
# print(type(test))
# print(test.ndim, test.shape)


plt.figure(figsize=(16,10))
plot_some(np.stack([restored_prob.mean(),restored_prob.scale()]), title_list=[['mean','scale']]);

plt.show()




# plt.figure(figsize=(16,10))
# plot_some(np.stack([x[3],restored,y[3]]),
#           title_list=[['low (maximum projection)','CARE (maximum projection)','GT (maximum projection)']],
#           pmin=2,pmax=99.8);
#
# plt.show()
