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

from tifffile import imread, imsave
from csbdeep.utils import plot_some
from csbdeep.models import CARE
# author Nicolas Russ (nicolas.riss22@gmail.com)

#x = imread("/home/nicolas/Bureau/MicroscopyImaging/Images/ruler_first_acquisition/pos3_50mw_1/pos3_50mw_1_MMStack_Pos0.ome.tif")
#x = imread("/home/nicolas/Bureau/MicroscopyImaging/Images/extract_spectra_beads_littleNoise.tif") #read an image as input
x = imread("/home/nicolas/Bureau/MicroscopyImaging/Images/extract_spectra_ruler.tif") #read an image as input

imageNumber = 29 #number of the image to read on the stack

modelName = "modelBeadsSpectra64_200pepoch_woBorder" #without noise
#modelName = "modelBeadsLittleNoiseSpectra64_200pepoch_woBorder" #noise added

model = CARE(config=None, name=modelName, basedir='models')
restored = model.predict(x[imageNumber], "YX", normalizer=None)

###################
# Show the result #
###################
plt.figure(figsize=(16,10))
plot_some(np.stack([x[imageNumber], restored]),
          title_list=[['source image','predicted (CARE)']],
          pmin=2,pmax=99.8);

plt.show()

#########################
# Predict probabilistic #
#########################
restored_prob = model.predict_probabilistic(x[imageNumber], "YX", normalizer=None) #axes?
plt.figure(figsize=(16,10))
plot_some(np.stack([restored_prob.mean(),restored_prob.scale()]), title_list=[['mean','scale']]);
plt.show()

######################
# Save the two files #
######################
imsave("restoredImage.tif", restored)
imsave("notRestoredImage.tif", x[imageNumber])

# CSBDeep has a save_tiff_imagej_compatible function that can be used also (TODO: study the difference)
# Sometimes, the images appears grey on ImageJ. It may be because of that.
# from csbdeep.io import save_tiff_imagej_compatible
# save_tiff_imagej_compatible('restoredImage.tif', restored, "YX")
# save_tiff_imagej_compatible('notRestoredImage.tif', x[imageNumber], "YX")
