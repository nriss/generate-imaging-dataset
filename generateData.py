from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
from createPatches.createPatches import createPatches
import sys

## Patch filter
np.set_printoptions(threshold=sys.maxsize) #writing entire data (not splitting) may consume time

##############################
# Localize spots and undrift #
##############################
'''
    Function Localize spots.
    This function uses a modified picasso library to identify and then return a hdf5 doc containing the position dot center for each frame.
    Then, the function undrifts the frames.
    https://github.com/jungmannlab/picasso
'''

def localizeSpots(path_X, path_Y):
    from localizeDots.localize import launchLocalize
    # Modified lib : picasso
    locsNet = launchLocalize(path_X)
    #print("locs : ", locsNet)
    locsFlou = launchLocalize(path_Y)
    #print("locs : ", locsFlou)


########################
# Identify common spots #
########################
def identifyDots(thresholdDistance, basepath, target_dir, source_dir, images_per_patches):
    from itertools import chain
    import h5py, math
    from operator import itemgetter
    try:
        from pathlib import Path
        Path().expanduser()
    except (ImportError,AttributeError):
        from pathlib2 import Path
    p = Path(basepath)
    pattern = '*.hdf5'
    pairs = [(f, p/target_dir/f.name) for f in chain(*((p/sd).glob(pattern) for sd in [source_dir]))]
    len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any hdf5 files containing spot localisation."))

    numberOfPointsUnderThreshold = 0
    result_array = []
    for fx, fy in pairs: #fx and fy are path to files x and y
        with h5py.File(fx, 'r') as f:
            with h5py.File(fy, 'r') as f:
                dataX = list(f['locs'])
                dataY = list(f['locs'])
                print(len(dataX), dataX[0])
                for dX in dataX:
                    for dY in dataY:
                        ####################
                        # dX and dY Format #
                        ####################
                        '''
                        @ see Localization HDF5 Files https://picassosr.readthedocs.io/en/latest/files.html#importing-hdf5-files-in-pandas-matlab-and-origin
                        0: frame number
                        1: x
                        2: y
                        3: photons
                        4: sx
                        5: sy
                        6:  bg
                        7: lpx
                        8: lpy
                        9: net_gradient
                        10: likelihood
                        11: Iterations
                        12(optional): group
                        13(optional): len
                        14(optional): n
                        15(optional): photon_rate
                        '''
                        Xposx = dX[1] # x position in image X
                        Xposy = dX[2]
                        Yposx = dY[1]
                        Yposy = dY[2]
                        dist = math.sqrt( (Xposx - Yposx) ** 2 + (Xposy - Yposy) ** 2 )
                        if (dist < thresholdDistance):
                            array = np.append(dX, dY)
                            array = np.array([dist, array], dtype=object)
                            numberOfPointsUnderThreshold += 1
                            result_array.append([dist, dX, dY])
                    if (len(result_array) > 10 * len(pairs) * images_per_patches): # we limit at ten times the number of spots we look at for computation optimization
                        break

    print("number of couple points under threshold found : ", numberOfPointsUnderThreshold)
    result_array = sorted(result_array, key=itemgetter(0))
    return result_array


#################
# GENERATE DATA #
#################
def generateData(basepath, source_dir, target_dir, n_patches_per_image, list_common_spots):
    from csbdeep.data import RawData, create_patches
    from csbdeep.io import save_training_data
    from csbdeep.data.generate import no_background_patches
    data = RawData.from_folder(basepath=basepath, source_dirs=[source_dir], target_dir=target_dir, axes='ZYX')
    #print(data.size)

    X, Y, XY_axes = createPatches(data, patch_size=(1, 64,128), n_patches_per_image=n_patches_per_image, verbose=True, list_common_spots=list_common_spots)
    #print(X, Y)

    print("shape of X,Y =", X.shape)
    print("axes  of X,Y =", XY_axes)

    save_training_data("test", X, Y, XY_axes)

    return X, Y, XY_axes

#############
# SHOW PLOT #
#############
'''
    function show plot
    show some example of X / Y patches
'''
def showPlot(X, Y, XY_axes):
    from csbdeep.utils import plot_some
    for i in range(2):
        plt.figure(figsize=(8,4))
        sl = slice(8*i, 8*(i+1)), 0
        plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start,sl[0].stop)])
        plt.show()


########
# MAIN #
########
basepath =  "data_ome_tif"
target_dir = "net"
source_dir = "flou"

thresholdDistance = 0.1 #in (sub)pixel
n_patches_per_image = 800


#localizeSpots(basepath + "/" + target_dir, basepath + "/" + source_dir)

list_common_spots = identifyDots(thresholdDistance, basepath, target_dir, source_dir, n_patches_per_image)

#createPatches(list_common_dots)
X, Y, XY_axes = generateData(basepath, target_dir, source_dir, n_patches_per_image, list_common_spots)
showPlot(X, Y, XY_axes)
