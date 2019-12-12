##############
# TEST MODEL #
##############
'''
    Python script used to test the model after computing
'''

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import math
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

from tifffile import imread, imsave
from csbdeep.utils import download_and_extract_zip_file, axes_dict, plot_some, plot_history
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE

#(X_train,Y_train), (X_val,Y_val), axes = load_training_data('test.npz', validation_split=0.1, verbose=True)

model = CARE(config=None, name='modelBeadsSpotsNoisy', basedir='.')

x = imread("/home/nicolas/Bureau/testImageReconstruction/1_20mw_1_MMStack_Pos0.ome.tif")

#x = imread("1_20mw_1_MMStack_Pos0.ome.tif")
restored = []
print("image numbers : ", len(x))
for i in range(0, len(x) - 1):
    if (i % 1000 == 0 and i / 1000 >= 1):
        print("Images restored: {}%, number of images: {}".format(math.trunc(i*100/len(x)), len(x)), end = '\r')
        imsave("restoredImage_1_20mw_1_MMStack_Pos0" + i + ".tif", restored)
        restored = []
    restored.append(model.predict(x[i], "YX", normalizer=None)) #axes?



# # print(type(test))
# # print(test.ndim, test.shape)
#
# from csbdeep.io import save_tiff_imagej_compatible
# save_tiff_imagej_compatible('testresult.tif', restored, "YX")
#
#
# plt.figure(figsize=(16,10))
# plot_some(np.stack([x[20],restored]),
#           title_list=[['low (maximum projection)','CARE (maximum projection)']],
#           pmin=2,pmax=99.8);
#
# plt.show()
#
# restored_prob = model.predict_probabilistic(x[20], "YX", normalizer=None) #axes?
# # print(type(test))
# # print(test.ndim, test.shape)
#
#
# plt.figure(figsize=(16,10))
# plot_some(np.stack([restored_prob.mean(),restored_prob.scale()]), title_list=[['mean','scale']]);
#
# plt.show()
#
#


# plt.figure(figsize=(16,10))
# plot_some(np.stack([x[3],restored,y[3]]),
#           title_list=[['low (maximum projection)','CARE (maximum projection)','GT (maximum projection)']],
#           pmin=2,pmax=99.8);
#
# plt.show()
