from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import sys

# uncomment ot write entire data (not splitting) may consume time
#np.set_printoptions(threshold=sys.maxsize)

##############################
# Localize spots and undrift #
##############################
'''
    Function Localize spots.
    This function uses the picasso library (modified).
    It returns an array with the localization of identified spots for each frame.
    the function launchLocalize also save a hdf5 file containing all informations.

    There is also an undrift function (have to be tested).
    https://github.com/jungmannlab/picasso
'''

def localizeSpots(path_X, path_Y):
    from localizeDots.localize import launchLocalize
    # Look for spots for tif files in the path_X directory
    locsTarget = launchLocalize(path_X)
    # Look for spots for tif files in the path_Y directory
    locsSource = launchLocalize(path_Y)

#########################
# Identify common spots #
#########################
'''
    Function identifySpots
    The goal of the function is to identify the spots which are present on two different images, under a threshold distance (in px).
    @param thresholdDistance : distance in px
    @param patchSize
    @param basepath : folder containing target_dir and source_dir folders with tif images
    @param target_dir : folder containing the target images
    @param source_dir : fonder containing the source images
    @param image_per_patches : number of images we want per patches (for perf optimization, we are looking for 10 * image_per_patches pairs of dots here).
    @param pathCommonSpots : filename to save common spot
'''
def identifySpots(thresholdDistance, patchSize, xDim, yDim, basepath, target_dir, source_dir, images_per_patches, pathCommonSpots):
    from itertools import chain
    import h5py, math
    from operator import itemgetter
    import math, random

    print("Common spots identification")

    try:
        from pathlib import Path
        Path().expanduser()
    except (ImportError,AttributeError):
        from pathlib2 import Path

    p = Path(basepath)
    pattern = '*.hdf5'
    pairs = [(f, p/target_dir/f.name) for f in chain(*((p/sd).glob(pattern) for sd in [source_dir]))]
    len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any hdf5 files containing spot localisation."))
    pairNumber = 0

    numberOfPointsUnderThreshold = 0
    result_array = {}
    for fx, fy in pairs: #fx and fy are path to files x and y
        print(fx, fy)
        pairNumber = pairNumber + 1
        print("------------------------------------------")
        print("--------- Processing pair {} of {}".format(pairNumber, len(pairs)), "---------")
        print("------------------------------------------")
        print()

        pairSet = []
        numberFound = 0
        with h5py.File(fx, 'r') as f:
            with h5py.File(fy, 'r') as g:
                print("1) Charging localization files")
                dataX = list(f['locs']) #taking so much time...
                dataY = list(g['locs']) #taking so much time...
                #################################
                # Prends du temps le random.... #
                #################################
                print("2) randomizing X images")
                dataX = sorted(dataX, key=lambda k: random.random()) # spot localization on X images

                print("3) Looking for common spots")
                indent = 0
                for dX in dataX:
                    Xposx = dX[1] # x position in image X
                    Xposy = dX[2]
                    indent+=1
                    ### Testing if we are not on the border of the image (will be deleted if we select dots everywhere on the screen)
                    if ((Xposx - thresholdDistance < (patchSize / 2)) or (Xposx + thresholdDistance > xDim - (patchSize / 2)) or (Xposy - thresholdDistance < (patchSize / 2)) or (Xposy + thresholdDistance > yDim - (patchSize / 2))):
                        continue;
                    print("Percentage done: {}%, pair number: {}".format(math.trunc(indent*100/len(dataX)), len(pairSet)), end = '\r')
                    xVal = [Xposx < b[1] + thresholdDistance and Xposx > b[1] - thresholdDistance for b in dataY]
                    for index, dY in enumerate(dataY):
                        if (not xVal[index]):
                            continue
                        ####################
                        # dX and dY Format #
                        ####################
                        '''
                        @see Localization HDF5 Files on https://picassosr.readthedocs.io/en/latest/files.html#importing-hdf5-files-in-pandas-matlab-and-origin
                        0: frame number
                        1: x
                        2: y
                        3: photons
                        4: sx
                        5: sy
                        6: bg
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
                        Yposx = dY[1]
                        Yposy = dY[2]

                        if (Yposy > Xposy + thresholdDistance or Yposy < Xposy - thresholdDistance):
                            # print("break")
                            continue
                        dist = math.sqrt( (Xposx - Yposx) ** 2 + (Xposy - Yposy) ** 2 )
                        if (dist < thresholdDistance): # Pair found !
                            array = np.array([dist, np.append(dX, dY)], dtype=object)
                            numberOfPointsUnderThreshold += 1
                            pairSet.append([dist, dX, dY])
                            #print("pair found : ", dist, dX, dY)
                            numberFound +=1
                            #TODO : look if there are other spots nearby
                            break;
                    if (numberFound >= 4 * images_per_patches): # we limit at ten times the number of spots we look at for computation optimization
                        print("Percentage done: {}%, pair number: {}".format(math.trunc(indent*100/len(dataX)), len(pairSet)), end = '\r')
                        print()
                        print("4) Ordering common spots per interest")
                        name = fx.absolute().as_posix().split('/')[-1].replace('_locs.hdf5', '')
                        print("name for dict : ", name)
                        pairSet = sorted(pairSet, key=lambda x: (x[2][3] + x[1][3]), reverse=True) #order by pixel intensity
                        #pairSet = sorted(pairSet, key=lambda x: (x[1][7] + x[1][8])/2) #order by localization precision on x/y mean

                        result_array[name] = pairSet
                        print()
                        break


    ####################
    ### SAVING SPOTs ###
    ####################
    import csv
    path = fx.absolute().as_posix()
    np.save(pathCommonSpots, np.asarray(result_array)) #can be improved ?

    print("number of couple points under threshold found : ", numberOfPointsUnderThreshold)
    print("result array len : ", len(result_array))
    return result_array


#########################################
# GENERATE DATA FOR CARE MODEL TRAINING #
#########################################
'''
    Function generateData
    The function will generate the patches and the npz file containing the patches, unsed to train the model.
    @param basepath : folder containing target_dir and source_dir folders with tif images
    @param target_dir : folder containing the target images
    @param source_dir : fonder containing the source images
    @param n_patches_per_image : number of patches per tif file.
    @param dict_common_spots : list of common spots obtained from localizeSpots function
    @param pathCommonSpots : filename to load common spot
'''
def generateData(basepath, target_dir, source_dir, n_patches_per_image, patchSize, dict_common_spots=None, pathCommonSpots=None):

    if (dict_common_spots == None and pathCommonSpots == None):
        print('/!\\ list_common_spot or fileName_common_spot is not defined in generateData, the pairSpots won\'t be taken into acccount')
    elif (dict_common_spots == None):
        dict_common_spots = np.load(pathCommonSpots, allow_pickle=True)

    from csbdeep.io import save_training_data
    from csbdeep.data.generate import no_background_patches
    from createPatches.createPatches import createPatches
    from createPatches.rawData import RawData
    data = RawData.from_folder(basepath=basepath, source_dirs=[source_dir], target_dir=target_dir)

    X, Y, XY_axes = createPatches(data, patch_size=(1, patchSize, patchSize), n_patches_per_image=n_patches_per_image, verbose=True, dict_common_spots=dict_common_spots)#, patch_axes="YX")
    print("shape of X,Y =", X.shape)
    print("axes  of X,Y =", XY_axes)

    return X, Y, XY_axes


def saveData(X, Y, XY_axes):
    from csbdeep.io import save_training_data
    save_training_data("test", X, Y, XY_axes)


#####################################
# Remove one dimension from patches #
#####################################
def removeFrameAxe(X, Y, XY_axes, patchSize):
    print("axes ", XY_axes)
    print("1 : ", X.ndim)
    print(X.shape[0], 1 , X.shape[3], X.shape[4])
    s = (X.shape[0], 1 , X.shape[3], X.shape[4])
    resultX = np.zeros(s)
    resultY = np.zeros(s)
    for  i, element in enumerate(X):
        resultX[i] = X[i][0]
        resultY[i] = Y[i][0]
    print("2 : ", resultX.ndim)
    print(resultX.shape)
    print(resultX.shape)
    return resultX, resultY, XY_axes.replace('Z', '')

#############
# SHOW PLOT #
#############
'''
    function show plot
    show examples of X / Y patches
'''
def showPlot(X, Y, XY_axes):
    from csbdeep.utils import plot_some
    for i in range(3):
        plt.figure(figsize=(8,4))
        sl = slice(8*i, 8*(i+1)), 0
        plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start,sl[0].stop)])
        plt.show()


########
# MAIN #
########

# Parameters
'''
    @param basepath : folder containing target_dir and source_dirs
    @param target_dir : target directory containing great quality images
    @param source_dirs : source directory containing poor quality images
    @param dict_common_spots : list of pairspots (saved in fileName_common_spot file)
    @param thresholdDistance : accepted distance between two spots of two images to consider them as the same spot
    @param n_patches_per_image : number of patches wanted per tif stack
    @param spotSize : patchSize in pixel
'''
basepath =  "data_20"
target_dir = "target"
source_dir = "source"
dict_common_spots = None
pathCommonSpots = basepath + "/commonSpots.npy"
#####################################
# Threshold distance in (sub)pixels #
# to consider two spots as the same #
#####################################
thresholdDistance = 0.1 #0.1 is great

###################################
# Number of patches for tif Files #
###################################
n_patches_per_image = 10
patchSize = 8

xDim = 512
yDim = 256


localizeSpots(basepath + "/" + target_dir, basepath + "/" + source_dir)

dict_common_spots = identifySpots(thresholdDistance, patchSize, xDim, yDim, basepath, target_dir, source_dir, n_patches_per_image, pathCommonSpots)

X, Y, XY_axes = generateData(basepath, target_dir, source_dir, n_patches_per_image, patchSize, dict_common_spots, pathCommonSpots)
X, Y, XY_axes = removeFrameAxe(X, Y, XY_axes, patchSize)

saveData(X, Y, XY_axes)

showPlot(X, Y, XY_axes)
