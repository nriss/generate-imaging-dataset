from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import sys

from csbdeep.io import save_training_data
from csbdeep.data.generate import no_background_patches
from createPatches.createPatches import createPatches
from createPatches.rawData import RawData
from csbdeep.utils import plot_some

#np.set_printoptions(threshold=sys.maxsize)
# uncomment ot write entire data (not splitting) may consume time

##############################
# Localize spots and undrift #
##############################
'''
    Function Localize spots.
    This function uses the picasso library (modified).
    It returns an array with the localization of identified spots for each frame.
    the function launchLocalize also save a hdf5 file containing all informations.

    There is also an undrift function (which have to be tested and implemented with this script).
    https://github.com/jungmannlab/picasso
'''

def localizeSpots(config):
    path_X = config['path']['basepath'] + "/" + config['path']['target_dir']
    path_Y = config['path']['basepath'] + "/" + config['path']['source_dir']
    from localizeDots.localize import launchLocalize
    # Look for spots for tif files in the path_X directory
    locsTarget = launchLocalize(path_X, config)
    # Look for spots for tif files in the path_Y directory
    locsSource = launchLocalize(path_Y, config)

#########################
# Identify common spots #
#########################
'''
    Function identifySpots
    The goal of the function is to identify the spots which are present on two different images, under a threshold distance (in px).
    @config : config object
    @config.thresholdDistance : distance in px
    @config.patchSize
    @config.target_dir : folder containing the target images
    @config.source_dir : fonder containing the source images
    @config.image_per_patches : number of images we want per patches (for perf optimization, we are looking for 10 * image_per_patches pairs of dots here).
    @config.pathCommonSpots : filename to save common spot
'''

def identifySpots(config):
    thresholdDistance = float(config['parameters']['thresholdDistance'])
    patchSize = int(config['parameters']['patchSize'])
    images_per_patches = int(config['parameters']['n_patches_per_image'])
    xDim = int(config['parameters']['xDim'])
    yDim = int(config['parameters']['yDim'])

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

    p = Path(config['path']['basepath'])
    pattern = '*.hdf5'
    pairs = [(f, p/config['path']['target_dir']/f.name) for f in chain(*((p/sd).glob(pattern) for sd in [config['path']['source_dir']]))]
    len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any hdf5 files containing spot localisation."))
    pairNumber = 0
    nearbyOffset = int(config['parameters']['offsetSpots']) # avoid that there are spots nearby
    numberOfPointsUnderThreshold = 0
    resultDict = {}
    for fx, fy in pairs: #fx and fy are path to files x and y
        pairNumber = pairNumber + 1
        print("------------------------------------------")
        print("--------- Processing pair {} of {}".format(pairNumber, len(pairs)), "---------")
        print("------------------------------------------")
        print()

        pairSet = []
        numberFound = 0
        with h5py.File(fx, 'r') as f:
            with h5py.File(fy, 'r') as g:
                done = False
                print("1) Charging localization files: ", fx)
                dataX = list(f['locs']) #taking so much time... X images
                dataY = list(g['locs']) #taking so much time... Y images

                print("2) randomizing X spots")
                dataX = sorted(dataX, key=lambda k: random.random()) # spot localization on X images

                print("3) Looking for common spots")
                indent = 0

                frameNumber = 0
                print("4) Creating X & Y spot dictionaries")
                ##############################################
                # creation of dictionaries ordered per frame #
                ##############################################
                dictYSpots = {}
                for i,e in enumerate(dataY):
                    frame = e[0]
                    if frame > frameNumber:
                        frameNumber = frame
                    try:
                        dictYSpots[str(frame)].append(e)
                    except KeyError:
                        dictYSpots[str(frame)] = []

                dictXSpots = {}
                for i,e in enumerate(dataX):
                    frame = e[0]
                    if frame > frameNumber:
                        frameNumber = frame
                    try:
                        dictXSpots[str(frame)].append(e)
                    except KeyError:
                        dictXSpots[str(frame)] = []

                for dX in dataX:
                    Xposx = dX[1] # x position in image X
                    Xposy = dX[2]
                    indent += 1

                    #######################################
                    # Not considering X spots near border # --> Avoid out of bound exceptions
                    #######################################
                    if config['parameters']['centralSpot'] == '1':
                        if ((Xposx - thresholdDistance < (patchSize / 2)) or (Xposx + thresholdDistance > xDim - (patchSize / 2)) or (Xposy - thresholdDistance < (patchSize / 2)) or (Xposy + thresholdDistance > yDim - (patchSize / 2))):
                            continue;

                    ######################################
                    # Verifying that the X spot is alone #
                    ######################################
                    if config['parameters']['multipleSpot'] == '0':
                        alone = True
                        if config['parameters']['centralSpot'] == '1':
                            # verif x
                            for sp in dictXSpots[str(dX[0])]:
                                if sp != dX:
                                    if abs(sp[1] - dX[1]) < (patchSize + nearbyOffset) / 2 and abs(sp[2] - dX[2]) < (patchSize + nearbyOffset) / 2:
                                        alone = False;
                                        break;
                            if not alone:
                                continue
                        if config['parameters']['centralSpot'] == '0':
                            basisX = (int(dX[1]) // patchSize) * patchSize
                            basisY = (int(dX[2]) // patchSize) * patchSize
                            for sp in dictXSpots[str(dX[0])]:
                                if sp != dX:
                                    if sp[1] - basisX  >= 0 - nearbyOffset and sp[1] - basisX <= patchSize + nearbyOffset and sp[2] - basisY >= 0 - nearbyOffset and sp[2] - basisY <= patchSize + nearbyOffset:
                                        alone = False;
                                        break;
                            if not alone:
                                continue
                            # TODO : with 0
                            # TODO : with 0 verif y
                            # TODO : with 0 verif x
                        #TO BE DONE

                    print("Percentage done: {}%, pair number: {}".format(math.trunc(indent*100/len(dataX)), len(pairSet)), end = '\r')
                    #xVal = [Xposx < b[1] + thresholdDistance and Xposx > b[1] - thresholdDistance for b in dataY]
                    exitFlag = False
                    for frame in range(0, frameNumber):
                        if exitFlag:
                            break; #a spot has already been found for this spot on X image
                        for dY in dictYSpots[str(frame)]:
                            ####################
                            # dX and dY Format #
                            ####################
                            '''
                            @see Localization HDF5 Files on https://picassosr.readthedocs.io/en/latest/files.html#importing-hdf5-files-in-pandas-matlab-and-origin
                            0: frame number
                            1,2: x,y (position)
                            3: photons
                            4,5: sx, sy (point spread function)
                            6: bg
                            7,8: lpx, lpy (localisation precision)
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

                            #######################################
                            # Not considering Y spots near border # --> Avoid out of bound exceptions
                            #######################################
                            if config['parameters']['centralSpot'] == '1':
                                if ((Yposx - thresholdDistance < (patchSize / 2)) or (Yposx + thresholdDistance > xDim - (patchSize / 2)) or (Yposy - thresholdDistance < (patchSize / 2)) or (Yposy + thresholdDistance > yDim - (patchSize / 2))):
                                    continue;

                            ####################################################
                            # Verifying that X and Y spots are not too distant # --> I think it is a little perf improvement
                            #################################################### --> avoid computing euclidean dist with sqrt for each spot
                            if (Yposy > Xposy + thresholdDistance or Yposy < Xposy - thresholdDistance):
                                continue

                            ##########################################
                            # Verifying distance between spots X & Y #
                            ##########################################
                            dist = math.sqrt( (Xposx - Yposx) ** 2 + (Xposy - Yposy) ** 2 )

                            if (dist < thresholdDistance): # Pair found !

                                ######################################
                                # Verifying that the Y spot is alone #
                                ######################################
                                if config['parameters']['multipleSpot'] == '0':
                                    if config['parameters']['centralSpot'] == '1':
                                        # verif y
                                        alone = True
                                        for sp in dictYSpots[str(frame)]:
                                            if sp != dY:
                                                if abs(sp[1] - dY[1]) < (patchSize + nearbyOffset) / 2 and abs(sp[2] - dY[2]) < (patchSize + nearbyOffset) / 2:
                                                    alone = False;
                                                    break;
                                    if config['parameters']['centralSpot'] == '0':
                                        basisX = (int(dY[1]) // patchSize) * patchSize
                                        basisY = (int(dY[2]) // patchSize) * patchSize
                                        for sp in dictYSpots[str(frame)]:
                                            if sp != dY:
                                                if sp[1] - basisX  >= 0 - nearbyOffset and sp[1] - basisX <= patchSize + nearbyOffset and sp[2] - basisY >= 0 - nearbyOffset and sp[2] - basisY <= patchSize + nearbyOffset:
                                                    alone = False;
                                                    break;
                                        if not alone:
                                            continue #ignoring this spot
                                    #if config['parameters']['centralSpot'] == '0':
                                        # TODO : with 0 verif y
                                    #TO BE DONE
                                numberOfPointsUnderThreshold += 1
                                pairSet.append([dist, dX, dY])
                                numberFound +=1
                                exitFlag = True
                                #TODO : look if there are other spots nearby
                                break;
                    if (numberFound >= images_per_patches * 2):
                        done = True
                        print("---> Number of pairs found : {} <---               ".format(len(pairSet)), end = '\r')
                        print()
                        #ordering
                        if (config['parameters']['spotOrder'] not in ["none", "None"]):
                            print("5) Ordering common spots per interest")
                        if (config['parameters']['spotOrder'] == 'intensity'):
                            pairSet = sorted(pairSet, key=lambda x: (x[2][3] + x[1][3]), reverse=True) #order by pixel intensity
                        elif(config['parameters']['spotOrder'] == 'lp'):
                            pairSet = sorted(pairSet, key=lambda x: (x[1][7] + x[1][8])/2) #order by localization precision on x/y mean

                        #saving
                        name = fx.absolute().as_posix().split('/')[-1].replace('_locs.hdf5', '').replace('.tif', '').replace('.ome', '')
                        resultDict[name] = pairSet
                        print()
                        break

                if (not done):
                    print("ERROR : NOT ENOUGH PATCHES FOUND")

    ####################
    ### SAVING SPOTs ###
    ####################
    from six.moves import cPickle as pickle #for performance

    with open(config['path']['commonSpots'], 'wb') as f:
        pickle.dump(resultDict, f, pickle.HIGHEST_PROTOCOL)

    print("number of couple points under threshold found : ", numberOfPointsUnderThreshold)
    return resultDict


#########################################
# GENERATE DATA FOR CARE MODEL TRAINING #
#########################################
'''
    Function generateData
    The function will generate the patches and the npz file containing the patches, unsed to train the model.
    @param config : config object
    @param config.target_dir : folder containing the target images
    @param config.source_dir : fonder containing the source images
    @param config.n_patches_per_image : number of patches per tif file.
    @param config.pathCommonSpots : filename to load common spot
    @param dict_common_spots : list of common spots obtained from localizeSpots function
'''
def generateData(config, dict_common_spots=None):

    if (dict_common_spots == None and config['path']['commonSpots'] == None):
        print('/!\\ list_common_spot or fileName_common_spot is not defined in generateData, the pairSpots won\'t be taken into acccount')
    elif (dict_common_spots == None):
        from six.moves import cPickle as pickle
        try:
            with open(config['path']['commonSpots'], 'rb') as f:
                dict_common_spots = pickle.load(f)
        except OSError as e:
            print("common spots file not found, common spots must be computed before generating data ! (Launch localizeSpots function)")

    data = RawData.from_folder(basepath=config['path']['basepath'], source_dirs=[config['path']['source_dir']], target_dir=config['path']['target_dir'])

    X, Y, XY_axes = createPatches(data, config, verbose=True, dict_common_spots=dict_common_spots)#, patch_axes="YX")
    print("shape of X,Y =", X.shape)
    print("axes  of X,Y =", XY_axes)

    return X, Y, XY_axes


def saveData(config, X, Y, XY_axes):
    ##################
    # Saving patches #
    ##################
    from csbdeep.io import save_training_data
    save_training_data(config['path']['basepath'] + "/patches", X, Y, XY_axes)

    ######################
    # Saving config data #
    ######################
    with open(config['path']['basepath'] + '/config', 'w') as configfile:
        config.write(configfile)


'''
    Function removeFrameAxe.
    This function removes the frame dimension of the patches to go from 3D to 2D patches.
'''
def removeFrameAxe(X, Y, XY_axes):
    s = (X.shape[0], 1 , X.shape[3], X.shape[4])
    resultX = np.zeros(s)
    resultY = np.zeros(s)
    for  i, element in enumerate(X):
        resultX[i] = X[i][0]
        resultY[i] = Y[i][0]

    #print(resultX.shape)
    return resultX, resultY, XY_axes.replace('Z', '')

#############
# SHOW PLOT #
#############
'''
    function show plot
    show examples of X / Y patches
'''
def showPlot(X, Y, XY_axes):
    for i in range(3):
        plt.figure(figsize=(8,4))
        sl = slice(5*i, 5*(i+1)), 0
        plot_some(X[sl], Y[sl], title_list=[np.arange(sl[0].start,sl[0].stop)]) #X puis Y
        plt.show()


#########################################################################################
#  _____        _                 _   __  __ _                                          #
# |  __ \      | |               | | |  \/  (_)                                         #
# | |  | | __ _| |_ __ _ ___  ___| |_| \  / |_  ___ _ __ ___  ___  ___ ___  _ __  _   _ #
# | |  | |/ _` | __/ _` / __|/ _ \ __| |\/| | |/ __| '__/ _ \/ __|/ __/ _ \| '_ \| | | |#
# | |__| | (_| | || (_| \__ \  __/ |_| |  | | | (__| | | (_) \__ \ (_| (_) | |_) | |_| |#
# |_____/ \__,_|\__\__,_|___/\___|\__|_|  |_|_|\___|_|  \___/|___/\___\___/| .__/ \__, |#
#                                                                          | |     __/ |#
#                   MAIN                                                   |_|    |___/ #
#########################################################################################

# Parameters
'''
    @param basepath : folder containing target_dir and source_dirs
    @param target_dir : target directory containing great quality images
    @param source_dirs : source directory containing poor quality images
    @param commonSpots :path to the common spot file for saving
    @param dict_common_spots : list of pairspots (saved in fileName_common_spot file)
    @param thresholdDistance : accepted distance between two spots of two images to consider them as the same spot
    @param n_patches_per_image : number of patches wanted per tif stack
    @param spotSize : patchSize in pixel
'''

import configparser
config = configparser.ConfigParser()
config['path'] = {'basepath': 'data_500',
                    'target_dir': 'target',
                    'source_dir': 'source'}
config['path']['commonSpots'] = config['path']['basepath'] + "/commonSpots"

config['parameters'] = {}


#######################
# Localize parameters #
#######################
# gradient parameter for localization, higer gradient need best defined spots to be considered
config['parameters']['localizeGradient'] = '7000'
# The threshold precision is the limit of acceptation of localisation precision of spots, in pixel, estimated by cramer-rao lower bound of the maximum likelihood fit
config['parameters']['thresholdPrecision'] = '0.3'


############################
# Generate data parameters #
############################
# Threshold distance in (sub)pixels to consider two spots as the same (0.1 is great)
config['parameters']['thresholdDistance'] = '0.3'
# Order the list of paired spots ?
config['parameters']['spotOrder'] = 'none' #possible value : 'intensity' / 'none'
# Authorize multiple spots on a patch ?
config['parameters']['multipleSpot'] = '0' #possible value : '1' for yes / '0' for not


######################
# Patches parameters #
######################
# number of patches extracted by image stack (min 20)
config['parameters']['n_patches_per_image'] = '75'
#patch size in px
config['parameters']['patchSize'] = '16'
# Would you like to centralize the spot in patches ? '0' for no, '1' for yes
config['parameters']['centralSpot'] = '0'
# avoid spots nerby the patch,
config['parameters']['offsetSpots'] = '4' #in px


####################
# Other parameters #
####################
#would be great to find dynamically xDim and yDim
config['parameters']['xDim'] = '512'
config['parameters']['yDim'] = '256'


#########
# DEBUG #
#########
config['parameters']['debugCentroid'] = '0' #place a black dot at the center of the spot (for debug purpose only)


list_common_spots = None

######################################
# 1) localization of spots (picasso) # parameters : thresholdPrecision, localizeGradient
######################################
localizeSpots(config) # taking i thresholdPrecision

############################################################
# 2) identification of common spots between the two stacks # Parameters : thresholdDistance, centralSpot
############################################################
list_common_spots = identifySpots(config) #modified by thresholdDistance

#######################
# 3) generate patches #
#######################
X, Y, XY_axes = generateData(config, list_common_spots)

#########################################
# 4) remove frame axe to have 2D images #
#########################################
X, Y, XY_axes = removeFrameAxe(X, Y, XY_axes)

###################################
# 5) Save patches on numpy format #
###################################
saveData(config, X, Y, XY_axes)

##################################
# 6) Show plots of paired images #
##################################
showPlot(X, Y, XY_axes)
