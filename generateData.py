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
    from localizeDots.localize import launchLocalize
    # Look for spots for tif files in the path_X directory
    path_X = config['path']['basepath'] + "/" + config['path']['source_dir']
    locsTarget = launchLocalize(path_X, config)
    # Look for spots for tif files in the path_Y directory
    path_Y = config['path']['basepath'] + "/" + config['path']['target_dir']
    locsSource = launchLocalize(path_Y, config)




#########################
# Identify common spots #
#########################
'''
    Function identifySpots
    The goal of the function is to identify the spots which are present on two different images, under a threshold distance (in px).
    @param config : config object
    @config.thresholdDistance : distance in px
    @config.patchSize
    @config.target_dir : folder containing the target images
    @config.source_dir : fonder containing the source images
    @config.image_per_patches : number of images we want per patches (for perf optimization, we are looking for 10 * image_per_patches pairs of dots here).
    @config.pathCommonSpots : filename to save common spot
    @param spectra : if true, avoid looking at spots with x < 75px to avoid learning on the transition data.
'''

def identifySpots(config, spectra):
    # Loading parameters
    thresholdDistance = float(config['parameters']['thresholdDistance'])
    patchSize = int(config['parameters']['patchSize'])
    patchSizeX = int(config['parameters']['patchSizeX'])

    images_per_patches = int(config['parameters']['n_patches_per_image'])
    xDim = int(config['parameters']['xDim'])
    yDim = int(config['parameters']['yDim'])

    ##############################################################
    # Only considering the spots, not taking account the spectra #
    ##############################################################
    XThreshold =  int(config['parameters']['XThreshold']) # Avoid out of bound exceptions


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

    ##########################################################
    # Loading hdf5 files containing localization information #
    ##########################################################
    p = Path(config['path']['basepath'])
    pattern = '*.hdf5'
    pairs = [(f, p/config['path']['target_dir']/f.name) for f in chain(*((p/sd).glob(pattern) for sd in [config['path']['source_dir']]))]
    len(pairs) > 0 or _raise(FileNotFoundError("Didn't find any hdf5 files containing spot localisation."))
    pairNumber = 0
    nearbyOffset = int(config['parameters']['offsetSpots']) # avoid that there are spots nearby
    numberOfPointsUnderThreshold = 0
    resultDict = {}

    for fx, fy in pairs: #fx and fy are path to files x (source) and y (target)
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

                print("2) randomizing X spots") #Avoid bias, we don't want to take spots from frame 1 in priority !
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
                        #####################################################################################
                        # Only considering the spots below the threshold, we don't want to consider spectra #
                        #####################################################################################
                        if e[1] < XThreshold: #--> Avoid out of bound exceptions
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

                for dX in dataX: # spots in source image, randomized order
                    Xposx = dX[1] # x position in image X
                    Xposy = dX[2]
                    indent += 1

                    #######################################
                    # Not considering X spots near border # --> Avoid out of bound exceptions
                    #######################################
                    if config['parameters']['centralSpot'] == '1':
                        # don't consider the spot near the border --> will raise an out of bound exception if we center spots
                        if ((Xposx - thresholdDistance < (patchSizeX / 2)) or (Xposx + thresholdDistance > XThreshold - (patchSizeX / 2)) or (Xposy - thresholdDistance < (patchSize / 2)) or (Xposy + thresholdDistance > yDim - (patchSize / 2))):
                            continue;

                    if spectra:
                        if (Xposx < 75):
                            #remove the transition between spetra and beads
                            #do not consider spots if the spectra can be in the transition area
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

                    print("Percentage done: {}%, pair number: {}".format(math.trunc(indent*100/len(dataX)), len(pairSet)), end = '\r')



                    exitFlag = False
                    for frame in random.shuffle(range(0, frameNumber)): #randomized frame in target image
                        if exitFlag:
                            break; #a spot has already been found for this spot on X image, avoid finding multiple patches for a same spot
                        for dY in random.shuffle(dictYSpots[str(frame)]): #randomized spots in target image
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
                            ####################################### --> Not needed because already verified before
                            # if config['parameters']['centralSpot'] == '1':
                            #     if ((Yposx - thresholdDistance < (patchSize / 2)) or (Yposx + thresholdDistance > xDim - (patchSize / 2)) or (Yposy - thresholdDistance < (patchSize / 2)) or (Yposy + thresholdDistance > yDim - (patchSize / 2))):
                            #         continue;

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
                                    alone = True
                                    if config['parameters']['centralSpot'] == '1':
                                        # verif y
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

                                numberOfPointsUnderThreshold += 1
                                pairSet.append([dist, dX, dY]) # distance, target, source
                                numberFound += 1
                                exitFlag = True
                                break;

                print("---> Number of pairs found : {} <---               ".format(len(pairSet)), end = '\r')
                print()

                #ordering
                if (config['parameters']['spotOrder'] not in ["none", "None"]):
                    print("5) Ordering common spots per interest")
                if (config['parameters']['spotOrder'] == 'intensity'):
                    #order by pixel intensity (target), could be interesting to get the most beautiful spectras
                    pairSet = sorted(pairSet, key=lambda x: x[1][3] + x[2][3], reverse=True)
                elif(config['parameters']['spotOrder'] == 'lp'):
                    pairSet = sorted(pairSet, key=lambda x: (x[1][7] + x[1][8])/2) #order by localization precision on x/y mean

                name = fx.absolute().as_posix().split('/')[-1].replace('_locs.hdf5', '').replace('.tif', '').replace('.ome', '')
                resultDict[name] = pairSet
                print()

                if (len(pairSet) < images_per_patches):
                    print("ERROR : NOT ENOUGH PATCHES FOUND  in ", name, "pair found : ", len(pairSet))
                    print("it can cause an error later, you should acquire new stacks with more common points or increase the thresholdDistance")
                    print()

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
    @param shifting : if we are looking for spectras, shift on ~200px on y axis to get the spectra
'''
def generateData(config, dict_common_spots=None, shifting=False):
    if (dict_common_spots == None and config['path']['commonSpots'] == None):
        print('/!\\ list_common_spot or fileName_common_spot is not defined in generateData, the pairSpots won\'t be taken into acccount')
    elif (dict_common_spots == None):
        from six.moves import cPickle as pickle
        try:
            with open(config['path']['commonSpots'], 'rb') as f:
                dict_common_spots = pickle.load(f)
        except OSError as e:
            print("common spots file not found, common spots must be computed before generating data ! (Launch localizeSpots function)")

    #Shift Y axis
    import copy
    dictCopy = copy.deepcopy(dict_common_spots)
    if (shifting):
        for key in dictCopy.keys():
            for i in range(0, len(dictCopy[key])):
                # shifting on y axis to get the spectra instead of spot
                dictCopy[key][i][1][1] =  dictCopy[key][i][1][1] + int(config['parameters']['shift'])
                dictCopy[key][i][2][1] =  dictCopy[key][i][2][1] + int(config['parameters']['shift'])


    data = RawData.from_folder(basepath=config['path']['basepath'], source_dirs=[config['path']['source_dir']], target_dir=config['path']['target_dir'])

    X, Y, XY_axes = createPatches(data, config, shifting, verbose=True, dict_common_spots=dictCopy)#, patch_axes="YX")
    print("shape of X,Y =", X.shape)
    print("axes  of X,Y =", XY_axes)

    return X, Y, XY_axes


def saveData(config, X, Y, XY_axes, spectra):
    ##################
    # Saving patches #
    ##################
    from csbdeep.io import save_training_data
    if (spectra):
        save_training_data(config['path']['patches'] + "_spectral", X, Y, XY_axes)
    else:
        save_training_data(config['path']['patches'], X, Y, XY_axes)

    ######################
    # Saving config data #
    ######################
    if (spectra):
        with open(config['path']['basepath'] + '/spectral_config', 'w') as configfile:
            config.write(configfile)
    else:
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
        plt.suptitle('5 example validation patches (top row: source, bottom row: target)');
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
config['path'] = {'basepath': 'data_ruler_final',
                    'target_dir': 'target',
                    'source_dir': 'source'}
config['path']['commonSpots'] = config['path']['basepath'] + "/commonSpotsShiftedForSpectra"
config['path']['patches'] = config['path']['basepath'] + "/patchesRuler"

config['parameters'] = {}


#######################
# Localize parameters #
#######################
# gradient parameter for localization,
# higher gradient need best defined spots to be considered
config['parameters']['localizeGradient'] = '5000'
# The threshold precision is the limit of acceptation of localisation precision of spots  (in px),
# estimated by cramer-rao lower bound of the maximum likelihood fit
config['parameters']['thresholdPrecision'] = '0.5'


############################
# Generate data parameters #
############################
# Threshold distance in (sub)pixels to consider two spots as the same (0.1 is great)
config['parameters']['thresholdDistance'] = '0.6'
# Order the list of paired spots ?
config['parameters']['spotOrder'] = 'intensity' #possible value : 'intensity' / 'none'.

# Authorize multiple spots on a patch ?
config['parameters']['multipleSpot'] = '1' #possible value : '1' for yes / '0' for not
# X threshold, under which the spots are, to avoid considering the spectral datas.
config['parameters']['XThreshold'] = '256' #in px
# shifts the x axis to the left to get the spectral data
config['parameters']['shift'] = '243' #in px


######################
# Patches parameters #
######################
# number of patches extracted by image stack (min 10)
config['parameters']['n_patches_per_image'] = '25'
#patch size in px
config['parameters']['patchSize'] = '16'
#patch size X is used for spectral patches (X are higher)
config['parameters']['patchSizeX'] = str(int(config['parameters']['patchSize']) * 4) #A specra is approx 79px.
# Would you like to centralize the spot in patches ? '0' for no, '1' for yes
config['parameters']['centralSpot'] = '0'
# avoid spots nearby the patch,
config['parameters']['offsetSpots'] = '1' #in px


####################
# Other parameters #
####################
#would be great to find dynamically xDim and yDim
#  xDim and yDim are used in generateData to see if spots are too close to the border
config['parameters']['xDim'] = '500'
config['parameters']['yDim'] = '200'


#########
# DEBUG #
#########
config['parameters']['debugCentroid'] = '0' #place a black dot at the center of the spot (for debug purpose only)

spectra = True

######################################
# 1) localization of spots (picasso) # parameters : thresholdPrecision, localizeGradient
######################################
#localizeSpots(config)

############################################################
# 2) identification of common spots between the two stacks # Parameters : thresholdDistance, centralSpot
############################################################
list_common_spots = None
list_common_spots = identifySpots(config, spectra) #modified by thresholdDistance

#######################
# 3) generate patches # # set the third parameter to True to get the spectra
#######################
X, Y, XY_axes = generateData(config, list_common_spots, spectra)
# X is the source patches
# Y is the target patches (high SNR)

#########################################
# 4) remove frame axe to have 2D images #
#########################################
X, Y, XY_axes = removeFrameAxe(X, Y, XY_axes)

###################################
# 5) Save patches on numpy format #
###################################
saveData(config, X, Y, XY_axes, spectra)

##################################
# 6) Show plots of paired images #
##################################
showPlot(X, Y, XY_axes)
