##############
# TEST MODEL #
##############
'''
    Python script used to test reconstruct a full stack of image
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
# author Nicolas Russ (nicolas.riss22@gmail.com)

#(X_train,Y_train), (X_val,Y_val), axes = load_training_data('test.npz', validation_split=0.1, verbose=True)

model = CARE(config=None, name='models/modelBeadsSpotsNoisy', basedir='.')

##############
# Read image #
##############
x = imread("/home/nicolas/Bureau/testImageReconstruction100/1_100mw_1_MMStack_Pos0.ome.tif")
y = imread("/home/nicolas/Bureau/testImageReconstruction100/1_100mw_1_MMStack_Pos0.ome.tif")

restored = []
print("image numbers : ", len(x))

##################
# Restore images #
##################
for i in range(0, len(x) - 1):
    if (i % 500 == 0):
        print("Images restored: {}%, number of images: {}".format(math.trunc(i*100/len(x)), len(x)), end = '\r')
    y[i] = model.predict(x[i], "YX", normalizer=None) #axes?

##############
# Save stack #
##############
imsave("restoredImage_1_100mw_1_MMStack_Pos0.tif", y)
