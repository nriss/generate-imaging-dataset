from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt

from tifffile import imread

from csbdeep.data import RawData, create_patches
from csbdeep.io import save_training_data
from csbdeep.data.generate import no_background_patches
from csbdeep.utils import plot_some


#################
# TRAINING DATA #
#################
data = RawData.from_folder(basepath='data_ome_tif', source_dirs=['flou'], target_dir='net', axes='ZYX')
print(data.size)

X, Y, XY_axes = create_patches(data, patch_size=(1, 64,128), n_patches_per_image=800, verbose=True,  patch_filter=no_background_patches(threshold=0.6, percentile=99.9))
#print(X, Y)

print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

save_training_data("test", X, Y, XY_axes)


for i in range(2):
    plt.figure(figsize=(8,4))
    sl = slice(8*i, 8*(i+1)), 0
    plot_some(X[sl],Y[sl],title_list=[np.arange(sl[0].start,sl[0].stop)])
    plt.show()
