from createPatches.transform import Transform, permute_axes, broadcast_target
from createPatches.utils import *

import numpy as np
import sys, os, warnings

from tqdm import tqdm
#Adapted from CSBDeep by Nicolas Riss

#((y,x), patch_size, n_patches_per_image, mask, patch_filter, dict_common_spots)

def sample_patches_from_multiple_stacks(datas, config, patch_size, n_samples, datas_mask=None, patch_filter=None, common_spots = None, verbose=False):
    """ sample matching patches of size `patch_size` from all arrays in `datas` """
    # TODO: some of these checks are already required in 'create_patches'
    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))

    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape,dtype=np.bool)
    else:
        patch_mask = patch_filter(datas, patch_size)

    if datas_mask is not None:
        # TODO: Test this
        warnings.warn('Using pixel masks for raw/transformed images not tested.')
        datas_mask.shape == datas[0].shape or _raise(ValueError())
        datas_mask.dtype == np.bool or _raise(ValueError())
        from scipy.ndimage.filters import minimum_filter
        patch_mask &= minimum_filter(datas_mask, patch_size, mode='constant', cval=False)
    # get the valid indices

    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[0].shape)])
    # print("patch_mask : ", patch_mask[border_slices])
    valid_inds = np.where(patch_mask[border_slices]) ## patch_mask[border_slices] contains mask

    if len(valid_inds[0]) == 0:
        raise ValueError("'patch_filter' didn't return any region to sample from")

    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]

    # sample
    sample_inds = np.random.choice(len(valid_inds[0]), n_samples, replace=len(valid_inds[0])<n_samples)

    rand_inds = [v[sample_inds] for v in valid_inds]

    if (common_spots is None): # Traditional way of CSBDeep
        res = [
                np.stack(
                    [
                        data[
                            tuple(
                                slice(
                                    _r - (_p//2),
                                    _r + _p-(_p//2)
                                )
                                for _r,_p in zip(r, patch_size)
                            )
                         ] for r in zip(*rand_inds)
                     ]
                    ) for data in datas
                ]
    else: #SR way with list of common spots between frames
        stackX = [] # source
        stackY = [] #target
        for commonSpot in common_spots:
            #distance = commonSpot[0]
            spot1 = commonSpot[1] #x
            spot2 = commonSpot[2] #y
            frame1 = commonSpot[1][0]
            frame2 = commonSpot[2][0]

            ############################################################################################
            #               Center on X but not on Y, to see the entire spectra on patches.            #
            # Would be interesting to randomize to add like +/- 50px to make the spectra position vary #
            ############################################################################################
            if config['parameters']['centralSpot'] == "2": #centered on X but not on Y, to see the entire spectra
                basisY = (int(spot1[2]) // patch_size[1]) * patch_size[1]

                valueX = [datas[1][frame1,
                                 basisY:basisY + patch_size[1],
                                 int(spot1[1]) - patch_size[2] // 2:int(spot1[1]) + patch_size[2] - patch_size[2] // 2,
                                 ]]
                valueY = [datas[0][frame2,
                                 basisY:basisY + patch_size[1],
                                 int(spot1[1]) - patch_size[2] // 2:int(spot1[1]) + patch_size[2] - patch_size[2] // 2,
                                 ]]

                if (config['parameters']['debugCentroid'] == '1'):
                    valueY[0][int(spot2[2]) % patch_size[1]][patch_size[2] // 2] = 0
                    valueX[0][int(spot1[2]) % patch_size[1]][patch_size[2] // 2] = 0

                stackY.append(valueY)
                stackX.append(valueX)

            elif (config['parameters']['centralSpot'] == '1'):
                valueX = [datas[1][frame1,
                                 int(spot1[2]) - patch_size[1] // 2:int(spot1[2]) + patch_size[1] - patch_size[1] // 2,
                                 int(spot1[1]) - patch_size[2] // 2:int(spot1[1]) + patch_size[2] - patch_size[2] // 2,
                                 ]]
                valueY = [datas[0][frame2,
                                 int(spot2[2]) - patch_size[1] // 2:int(spot2[2]) + patch_size[1] - patch_size[1] // 2,
                                 int(spot2[1]) - patch_size[2] // 2:int(spot2[1]) + patch_size[2] - patch_size[2] // 2,
                                 ]]

                if (config['parameters']['debugCentroid'] == '1'):
                    valueY[0][patch_size[1] // 2][patch_size[2] // 2] = 0
                    valueX[0][patch_size[1] // 2][patch_size[2] // 2] = 0

                stackY.append(valueY)
                stackX.append(valueX)

            else:
                basisY = (int(spot1[2]) // patch_size[1]) * patch_size[1]
                basisX = (int(spot1[1]) // patch_size[2]) * patch_size[2]
                valueX = [datas[1][frame1,
                                 basisY:basisY + patch_size[1],
                                 basisX:basisX + patch_size[2],
                                 ]]
                valueY = [datas[0][frame2,
                                 basisY:basisY + patch_size[1],
                                 basisX:basisX + patch_size[2],
                                 ]]

                if (config['parameters']['debugCentroid'] == '1'):
                    valueY[0][int(spot2[2]) % patch_size[1]][int(spot2[1]) % patch_size[2]] = 0
                    valueX[0][int(spot1[2]) % patch_size[1]][int(spot1[1]) % patch_size[2]] = 0

                stackY.append(valueY)
                stackX.append(valueX)
            ## extract image part
            if (len(stackX) == n_samples): # stopping when we have enough images
                break;

        res = [np.stack([x]) for x in [stackY, stackX]]

    return res





def no_background_patches(threshold=0.4, percentile=99.9):

    """Returns a patch filter to be used by :func:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain a substantial amount of non-background signal. To that end, a maximum filter is applied to the target image
    to find the largest values in a region.

    Parameters
    ----------
    threshold : float, optional
        Scalar threshold between 0 and 1 that will be multiplied with the (outlier-robust)
        maximum of the image (see `percentile` below) to denote a lower bound.
        Only patches with a maximum value above this lower bound are eligible to be sampled.
    percentile : float, optional
        Percentile value to denote the (outlier-robust) maximum of an image, i.e. should be close 100.

    Returns
    -------
    function
        Function that takes an image pair `(y,x)` and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :func:`create_patches`). At least one pixel of the
        binary mask must be ``True``, otherwise there are no patches to sample.

    Raises
    ------
    ValueError
        Illegal arguments.
    """

    (np.isscalar(percentile) and 0 <= percentile <= 100) or _raise(ValueError())
    (np.isscalar(threshold)  and 0 <= threshold  <=   1) or _raise(ValueError())

    from scipy.ndimage.filters import maximum_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        filtered = maximum_filter(image, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image,percentile)
    return _filter


def sample_percentiles(pmin=(1,3), pmax=(99.5,99.9)):
    """Sample percentile values from a uniform distribution.

    Parameters
    ----------
    pmin : tuple
        Tuple of two values that denotes the interval for sampling low percentiles.
    pmax : tuple
        Tuple of two values that denotes the interval for sampling high percentiles.

    Returns
    -------
    function
        Function without arguments that returns `(pl,ph)`, where `pl` (`ph`) is a sampled low (high) percentile.

    Raises
    ------
    ValueError
        Illegal arguments.
    """
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1] < pmax[0] or _raise(ValueError())
    return lambda: (np.random.uniform(*pmin), np.random.uniform(*pmax))

def _valid_low_high_percentiles(ps):
    return isinstance(ps,(list,tuple,np.ndarray)) and len(ps)==2 and all(map(np.isscalar,ps)) and (0<=ps[0]<ps[1]<=100)


def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024*1024**2):
    try:
        # raise ImportError
        import psutil
        mem = psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.available
        if mem_frac > 1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac > thresh_free_frac:
            print('Warning: will use at least %.0f MB (%.1f%%) of available memory.\n' % (n_required_memory_bytes/1024**2,100*mem_frac), file=sys.stderr)
            sys.stderr.flush()
    except ImportError:
        if n_required_memory_bytes > thresh_abs_bytes:
            print('Warning: will use at least %.0f MB of memory.\n' % (n_required_memory_bytes/1024**2), file=sys.stderr)
            sys.stderr.flush()



def norm_percentiles(percentiles=sample_percentiles(), relu_last=False):
    """Normalize extracted patches based on percentiles from corresponding raw image.

    Parameters
    ----------
    percentiles : tuple, optional
        A tuple (`pmin`, `pmax`) or a function that returns such a tuple, where the extracted patches
        are (affinely) normalized in such that a value of 0 (1) corresponds to the `pmin`-th (`pmax`-th) percentile
        of the raw image (default: :func:`sample_percentiles`).
    relu_last : bool, optional
        Flag to indicate whether the last activation of the CARE network is/will be using
        a ReLU activation function (default: ``False``)

    Return
    ------
    function
        Function that does percentile-based normalization to be used in :func:`create_patches`.

    Raises
    ------
    ValueError
        Illegal arguments.

    Todo
    ----
    ``relu_last`` flag problematic/inelegant.

    """
    if callable(percentiles):
        _tmp = percentiles()
        _valid_low_high_percentiles(_tmp) or _raise(ValueError(_tmp))
        get_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        get_percentiles = lambda: percentiles

    def _normalize(patches_x,patches_y, x,y,mask,channel):
        pmins, pmaxs = zip(*(get_percentiles() for _ in patches_x))
        percentile_axes = None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a,p: np.percentile(a,p,axis=percentile_axes,keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x, _perc(x,pmins), _perc(x,pmaxs))
        if relu_last:
            pmins = np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y,pmins), _perc(y,pmaxs))
        return patches_x_norm, patches_y_norm

    return _normalize

def shuffle_inplace(*arrs,**kwargs):
    seed = kwargs.pop('seed', None)
    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed=seed)
    state = rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)

'''
Added list common spots to select only common spots.
'''
def createPatches(
        raw_data,
        config,
        dict_common_spots = None,
        patch_axes    = None,
        save_file     = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        normalization = norm_percentiles(),
        shuffle       = True,
        verbose       = True,
    ):
    """Create normalized training data to be used for neural network training.

    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    config : Config object
        list of configuration objects for generateData
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.

    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images, `n_channels` will be 1.

    Raises
    ------
    ValueError
        Various reasons.

    Example
    -------
    >>> raw_data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> X, Y, XY_axes = create_patches(raw_data, patch_size=(32,128,128), n_patches_per_image=16)

    Todo
    ----
    - Save created patches directly to disk using :class:`numpy.memmap` or similar?
      Would allow to work with large data that doesn't fit in memory.

    """
    if (config['parameters']['Spectra'] == "1"):
        patch_size = (1, int(config['parameters']['patchSize']), int(config['parameters']['patchSizeX']))
    else:
        patch_size = (1, int(config['parameters']['patchSize']), int(config['parameters']['patchSize']))
    n_patches_per_image = int(config['parameters']['n_patches_per_image'])


    ## images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())


    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches * np.prod(patch_size) * 4

    ## memory check
    _memory_check(n_required_memory_bytes)

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)

    sys.stdout.flush()

    ## sample patches from each pair of transformed raw images
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)

    for i, (x,y,_axes,mask, pathx, pathy) in tqdm(enumerate(image_pairs),total=n_images,disable=(not verbose)):
        if i >= n_images:
            warnings.warn('more raw images (or transformations thereof) than expected, skipping excess images.')
            break
        if i==0:
            axes = axes_check_and_normalize(_axes,len(patch_size))
            channel = axes_dict(axes)['C']
        # checks
        # len(axes) >= x.ndim or _raise(ValueError())
        axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))

        name = pathx.absolute().as_posix().split('/')[-1].replace('_locs.hdf5', '').replace('.tif', '').replace('.ome', '')
        _Y,_X = sample_patches_from_multiple_stacks((y,x), config, patch_size, n_patches_per_image, mask, patch_filter, dict_common_spots[name])

        s = slice(i*n_patches_per_image,(i+1) * n_patches_per_image)

        X[s], Y[s] = normalization(_X, _Y, x, y, mask, channel) # if error here : not enough patches found, you have to create lower patches or lower the localize gradient

    if shuffle:
        shuffle_inplace(X,Y)

    axes = 'SC'+axes.replace('C','')
    if channel is None:
        X = np.expand_dims(X,1)
        Y = np.expand_dims(Y,1)
    else:
        X = np.moveaxis(X, 1+channel, 1)
        Y = np.moveaxis(Y, 1+channel, 1)

    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, X, Y, axes)

    return X,Y,axes
