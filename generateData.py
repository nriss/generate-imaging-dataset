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
    @param fileName_common_spot : filename to save common spot
'''
def identifySpots(thresholdDistance, patchSize, xDim, yDim, basepath, target_dir, source_dir, images_per_patches):
    from itertools import chain
    import h5py, math
    from operator import itemgetter
    import math, random

    print("identification of spots")
    #TODO : data from identifySpots can be loaded in generateData if already computed

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
    result_array = []
    for fx, fy in pairs: #fx and fy are path to files x and y
        pairNumber = pairNumber + 1
        print("------------------------------------------")
        print("--- Processing pair {} of {}".format(pairNumber, len(pairs)))
        print("------------------------------------------")

        pairSet = []
        numberFound = 0
        with h5py.File(fx, 'r') as f:
            with h5py.File(fy, 'r') as g:
                print("charging 1")
                dataX = list(f['locs']) #taking so much time...
                print("charging 2")
                dataY = list(g['locs']) #taking so much time...
                #################################
                # Prends du temps le random.... #
                #################################
                print("randomizing")
                dataX = sorted(dataX, key=lambda k: random.random()) # spot localization on X images
                #dataY = sorted(dataY, key=lambda k: random.random()) # spot localization on Y images
                indent = 0
                for dX in dataX:
                    print("Percentage done: {}%, pair number: {}".format(math.trunc(indent*100/len(dataX)), len(pairSet)), end = '')
                    indent+=1
                    for dY in dataY:
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
                        Xposx = dX[1] # x position in image X
                        Xposy = dX[2]
                        Yposx = dY[1]
                        Yposy = dY[2]
                        if (Yposx > Xposx):
                            break;
                        dist = math.sqrt( (Xposx - Yposx) ** 2 + (Xposy - Yposy) ** 2 )
                        if (dist < thresholdDistance):
                            if (Xposx - thresholdDistance > (patchSize / 2) and Xposx + thresholdDistance < xDim - (patchSize / 2)):
                                if (Xposy - thresholdDistance > (patchSize / 2) and Xposy + thresholdDistance < yDim - (patchSize / 2)):
                                    array = np.append(dX, dY)
                                    array = np.array([dist, array], dtype=object)
                                    numberOfPointsUnderThreshold += 1
                                    pairSet.append([dist, dX, dY])
                                    #print(dist, dX, dY)
                                    numberFound +=1
                            #TODO : look if there are other spots nearby
                    if (numberFound > 10 * images_per_patches): # we limit at ten times the number of spots we look at for computation optimization
                        #TODO: function to save the spots
                        break

        ####################
        ### SAVING SPOTs ###
        ####################
        import csv

        np.save(fx.replace(".hdf5", "_pairSpots.npy").replace(target_dir, "").replace(source_dir, ""), np.asarray(result_array)) #can be improved ?

        pairSet = sorted(pairSet, key=lambda x: random.random())
        result_array.append(pairSet)

    print("number of couple points under threshold found : ", numberOfPointsUnderThreshold)
    #result_array = sorted(result_array, key=lambda x: x[1][3], reverse=True) #order by pixel intensity


    # for i, e in enumerate(result_array):
    #     print(e[2])
    # #print(result_array)

    #numberOfPointsUnderThreshold = random.shuffle(numberOfPointsUnderThreshold)

    #
    # with open("test.txt", "w") as txt_file:
    #     for line in result_array:
    #         txt_file.write(" ".join(str(line)) + "\n") # works with any number of elements in a line
    # contains list of [distance, [spot 1], [spot 2]]

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
    @param list_common_spots : list of common spots obtained from localizeSpots function
    @param fileName_common_spot : filename to load common spot
'''
def generateData(basepath, target_dir, source_dir, n_patches_per_image, patchSize, list_common_spots=None, fileName_common_spot=None):

    if (list_common_spots == None and fileName_common_spot == None):
        print('/!\\ list_common_spot or fileName_common_spot is not defined in generateData, the pairSpots won\'t be taken into acccount')
    elif (list_common_spots == None):
        list_common_spots = np.load(fileName_common_spot, allow_pickle=True)

    from csbdeep.io import save_training_data
    from csbdeep.data.generate import no_background_patches
    from createPatches.createPatches import createPatches
    from createPatches.rawData import RawData
    data = RawData.from_folder(basepath=basepath, source_dirs=[source_dir], target_dir=target_dir)

    X, Y, XY_axes = createPatches(data, patch_size=(1, patchSize, patchSize), n_patches_per_image=n_patches_per_image, verbose=True, list_common_spots=list_common_spots)#, patch_axes="YX")
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
    for i in range(2):
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
    @param list_common_spots : list of pairspots (saved in fileName_common_spot file)
    @param thresholdDistance : accepted distance between two spots of two images to consider them as the same spot
    @param n_patches_per_image : number of patches wanted per tif stack
    @param spotSize : patchSize in pixel
'''
basepath =  "data"
target_dir = "target"
source_dir = "source"
list_common_spots = None
#####################################
# Threshold distance in (sub)pixels #
# to consider two spots as the same #
#####################################
thresholdDistance = 0.1

###################################
# Number of patches for tif Files #
###################################
n_patches_per_image = 1000
patchSize = 32

xDim = 512
yDim = 256
#localizeSpots(basepath + "/" + target_dir, basepath + "/" + source_dir)

list_common_spots = identifySpots(thresholdDistance, patchSize, xDim, yDim, basepath, target_dir, source_dir, n_patches_per_image)
#
# X, Y, XY_axes = generateData(basepath, target_dir, source_dir, n_patches_per_image, patchSize, list_common_spots, fileName_common_spot)
# X, Y, XY_axes = removeFrameAxe(X, Y, XY_axes, patchSize)
#
# saveData(X, Y, XY_axes)
#
# showPlot(X, Y, XY_axes)
