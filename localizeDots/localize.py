#!/usr/bin/env python


"""
    picasso.localize
    ~~~~~~~~~~~~~~~~

    Identify and localize fluorescent single molecules in a frame sequence

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    :copyright: Copyright (c) 2016-2018 Jungmann Lab, MPI of Biochemistry
"""
import numpy as _np
import numba as _numba
import multiprocessing as _multiprocessing
import ctypes as _ctypes
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
import threading as _threading
from itertools import chain as _chain
import matplotlib.pyplot as _plt
import localizeDots.gaussmle as _gaussmle
import localizeDots.ioLocalize as _io
import localizeDots.postprocess as postprocess
from localizeDots.ioLocalize import load_movie, save_locs, save_info, load_locs


_C_FLOAT_POINTER = _ctypes.POINTER(_ctypes.c_float)
LOCS_DTYPE = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("net_gradient", "f4"),
    ("likelihood", "f4"),
    ("iterations", "i4"),
]


_plt.style.use("ggplot")


@_numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame, box):
    """ Finds pixels with maximum value within a region of interest """
    Y, X = frame.shape
    maxima_map = _np.zeros(frame.shape, _np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half: i + box_half + 1,
                j - box_half: j + box_half + 1,
            ]
            flat_max = _np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = _np.where(maxima_map)
    return y, x


@_numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@_numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame, y, x, box, uy, ux):
    box_half = int(box / 2)
    ng = _np.zeros(len(x), dtype=_np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(
                range(xi - box_half, xi + box_half + 1)
            ):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += (
                        gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
                    )
    return ng


@_numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image, minimum_ng, box):
    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = _np.zeros((box, box), dtype=_np.float32)
    uy = _np.zeros((box, box), dtype=_np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = _np.sqrt(ux ** 2 + uy ** 2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(frame, minimum_ng, box, roi=None):
    if roi is not None:
        frame = frame[roi[0][0]: roi[1][0], roi[0][1]: roi[1][1]]
    image = _np.float32(frame)  # otherwise numba goes crazy
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x, net_gradient


def identify_by_frame_number(movie, minimum_ng, box, frame_number, roi=None):
    frame = movie[frame_number]
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * _np.ones(len(x))
    return _np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[("frame", "i"), ("x", "i"), ("y", "i"), ("net_gradient", "f4")],
    )


def _identify_worker(movie, current, minimum_ng, box, roi, lock):
    n_frames = len(movie)
    identifications = []
    while True:
        with lock:
            index = current[0]
            if index == n_frames:
                return identifications
            current[0] += 1
        identifications.append(
            identify_by_frame_number(movie, minimum_ng, box, index, roi)
        )
    return identifications


def identifications_from_futures(futures):
    ids_list_of_lists = [_.result() for _ in futures]
    ids_list = _chain(*ids_list_of_lists)
    ids = _np.hstack(ids_list).view(_np.recarray)
    ids.sort(kind="mergesort", order="frame")
    return ids


def identify_async(movie, minimum_ng, box, roi=None):
    "Use the user settings to define the number of workers that are being used"
    settings = _io.load_user_settings()
    try:
        cpu_utilization = settings["Localize"]["cpu_utilization"]
        if cpu_utilization >= 1:
            cpu_utilization = 1
    except Exception as e:
        print(e)
        print(
            "An Error occured. Setting cpu_utilization to 0.8"
        )  # TODO at some point re-write this
        cpu_utilization = 0.8
        settings["Localize"]["cpu_utilization"] = cpu_utilization
        _io.save_user_settings(settings)

    n_workers = max(1, int(cpu_utilization * _multiprocessing.cpu_count()))

    current = [0]
    executor = _ThreadPoolExecutor(n_workers)
    lock = _threading.Lock()
    f = [
        executor.submit(
            _identify_worker, movie, current, minimum_ng, box, roi, lock
        )
        for _ in range(n_workers)
    ]
    executor.shutdown(wait=False)
    return current, f


def identify(movie, minimum_ng, box, threaded=True):
    if threaded:
        current, futures = identify_async(movie, minimum_ng, box)
        identifications = [_.result() for _ in futures]
        identifications = [_np.hstack(_) for _ in identifications]
    else:
        identifications = [
            identify_by_frame_number(movie, minimum_ng, box, i)
            for i in range(len(movie))
        ]
    return _np.hstack(identifications).view(_np.recarray)


@_numba.jit(nopython=True, cache=False)
def _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box):
    n_spots = len(ids_x)
    r = int(box / 2)
    spots = _np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r: yc + r + 1, xc - r: xc + r + 1]
    return spots


@_numba.jit(nopython=True, cache=False)
def _cut_spots_frame(
    frame, frame_number, ids_frame, ids_x, ids_y, r, start, N, spots
):
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r: yc + r + 1, xc - r: xc + r + 1]
    return j


def _cut_spots(movie, ids, box):
    if isinstance(movie, _np.ndarray):
        return _cut_spots_numba(movie, ids.frame, ids.x, ids.y, box)
    else:
        """ Assumes that identifications are in order of frames! """
        r = int(box / 2)
        N = len(ids.frame)
        spots = _np.zeros((N, box, box), dtype=movie.dtype)
        start = 0
        for frame_number, frame in enumerate(movie):
            start = _cut_spots_frame(
                frame,
                frame_number,
                ids.frame,
                ids.x,
                ids.y,
                r,
                start,
                N,
                spots,
            )
        return spots


def _to_photons(spots, camera_info):
    spots = _np.float32(spots)
    baseline = camera_info["baseline"]
    sensitivity = camera_info["sensitivity"]
    gain = camera_info["gain"]
    qe = camera_info["qe"]
    return (spots - baseline) * sensitivity / (gain * qe)


def get_spots(movie, identifications, box, camera_info):
    spots = _cut_spots(movie, identifications, box)
    return _to_photons(spots, camera_info)


def fit(
    movie,
    camera_info,
    identifications,
    box,
    eps=0.001,
    max_it=100,
    method="sigma",
):
    spots = get_spots(movie, identifications, box, camera_info)
    theta, CRLBs, likelihoods, iterations = _gaussmle.gaussmle(
        spots, eps, max_it, method=method
    )
    return locs_from_fits(
        identifications, theta, CRLBs, likelihoods, iterations, box
    )


def fit_async(
    movie,
    camera_info,
    identifications,
    box,
    eps=0.001,
    max_it=100,
    method="sigma",
):
    spots = get_spots(movie, identifications, box, camera_info)
    return _gaussmle.gaussmle_async(spots, eps, max_it, method=method)


def locs_from_fits(
    identifications, theta, CRLBs, likelihoods, iterations, box, config=None
):

    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    #with _np.errstate(invalid="ignore"):

    # BEGINING modif
    # Modified by Nicolas Riss because caused an error (trying to do sqrt on negative values) --> replacing negatives by 0
    for i in range (len(CRLBs)):
        CRLBs[i][CRLBs[i] < 0] = 0
    # END modif

    lpy = _np.sqrt(CRLBs[:, 0])
    lpx = _np.sqrt(CRLBs[:, 1])
    #for index, value in identifications.frame:
    locs = _np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 2],
            theta[:, 5],
            theta[:, 4],
            theta[:, 3],
            lpx,
            lpy,
            identifications.net_gradient,
            likelihoods,
            iterations,
        ),
        dtype=LOCS_DTYPE,
    )
    if config != None:
        locs = locs[ (locs['lpx'] < float(config['parameters']['thresholdPrecision']) ) ]
        locs = locs[ (locs['lpy'] < float(config['parameters']['thresholdPrecision']) ) ]
    print("Conserving {} spots under localization precision threshold".format(len(locs)))#, len(locs[locs['lpx'] < ]))
    locs.sort(kind="mergesort", order="frame")
    return locs


def localize(movie, info, parameters):
    print("localizing")
    identifications = identify(movie, parameters)
    return fit(movie, info, identifications, parameters["Box Size"])




"""
    ..__main__.py
    ~~~~~~~~~~~~~~~~

    Picasso command line interface

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss
    :copyright: Copyright (c) 2016-2019 Jungmann Lab, MPI of Biochemistry
"""
import os.path



def _undrift(files, segmentation, display=True, fromfile=None):
    import glob
    from numpy import genfromtxt, savetxt

    paths = glob.glob(files)
    undrift_info = {"Generated by": "Picasso Undrift"}
    if fromfile is not None:
        undrift_info["From File"] = fromfile
        drift = genfromtxt(fromfile)
    else:
        undrift_info["Segmentation"] = segmentation
    for path in paths:
        locs, info = load_locs(path)
        info.append(undrift_info)
        if fromfile is not None:
            # this works for mingjies drift files but not for the own ones
            locs.x -= drift[:, 1][locs.frame]
            locs.y -= drift[:, 0][locs.frame]
        else:
            print("Undrifting file {}".format(path))
            drift, locs = postprocess.undrift(
                locs, info, segmentation, display=display
            )
        base, ext = os.path.splitext(path)
        io.save_locs(base + "_undrift.hdf5", locs, info)
        savetxt(base + "_drift.txt", drift, header="dx\tdy", newline="\r\n")






def _localize(args, config):
    result = []
    from glob import glob

    from os.path import splitext, isdir
    from time import sleep
    import os.path as _ospath
    import re as _re
    import os as _os
    import yaml as yaml

    files = args.files

    print("    ____  _____________   __________ ____ ")
    print("   / __ \\/  _/ ____/   | / ___/ ___// __ \\")
    print("  / /_/ // // /   / /| | \\__ \\\\__ \\/ / / /")
    print(" / _____/ // /___/ ___ |___/ ___/ / /_/ / ")
    print("/_/   /___/\\____/_/  |_/____/____/\\____/  ")
    print("                                          ")
    print("------------------------------------------")
    print("Localize - Parameters:")
    print("{:<8} {:<15} {:<10}".format("No", "Label", "Value"))

    if args.fit_method == "lq-gpu":
        if gausslq.gpufit_installed:
            print("GPUfit installed")
        else:
            raise Exception("GPUfit not installed. Aborting.")

    for index, element in enumerate(vars(args)):
        print(
            "{:<8} {:<15} {:<10}".format(
                index + 1, element, getattr(args, element)
            )
        )
    print("------------------------------------------")

    # def check_consecutive_tif(filepath):
    #     """
    #     Function to only return the first file of a consecutive ome.tif series
    #     to not reconstruct all of them as load_movie automatically detects
    #     consecutive files. E.g. have a folder with file.ome.tif,
    #     file_1.ome.tif, file_2.ome.tif, will return only file.ome.tif
    #     """
    #     files = glob(filepath + "/*.tif")
    #     newlist = [os.path.abspath(file) for file in files]
    #     for file in files:
    #         path = os.path.abspath(file)
    #         directory = os.path.dirname(path)
    #         base, ext = os.path.splitext(
    #             os.path.splitext(path)[0]
    #         )  # split two extensions as in .ome.tif
    #         base = _re.escape(base)
    #         pattern = _re.compile(
    #             base + r"_(\d*).ome.tif"
    #         )  # This matches the basename + an appendix of the file number
    #         entries = [_.path for _ in os.scandir(directory) if _.is_file()]
    #         matches = [_re.match(pattern, _) for _ in entries]
    #         matches = [_ for _ in matches if _ is not None]
    #         datafiles = [_.group(0) for _ in matches]
    #         if datafiles != []:
    #             for element in datafiles:
    #                 newlist.remove(element)
    #     return newlist

    if os.path.isdir(files):
        print("Analyzing folder")

        f = glob(files + "/*.tif")
        tif_files = [os.path.abspath(file) for file in f]
        #tif_files = check_consecutive_tif(files)

        paths = tif_files + glob(files + "/*.raw")
        print("A total of {} files detected".format(len(paths)))
    else:
        paths = glob(files)

    # Check for raw files: make sure that each contains a yaml file
    def prompt_info():
        info = {}
        info["Byte Order"] = input("Byte Order (< or >): ")
        info["Data Type"] = input('Data Type (e.g. "uint16"): ')
        info["Frames"] = int(input("Frames: "))
        info["Height"] = int(input("Height: "))
        info["Width"] = int(input("Width: "))
        save = input("Use for all remaining raw files in folder (y/n)?") == "y"
        return info, save

    save = False
    for path in paths:
        base, ext = os.path.splitext(path)
        if ext == '.raw':
            if not os.path.isfile(base+'.yaml'):
                print('No yaml found for {}. Please enter:'.format(path))
                if not save:
                    info, save = prompt_info()
                info_path = base+'.yaml'
                save_info(info_path, [info])

    if paths:
        box = args.box_side_length
        min_net_gradient = args.gradient
        camera_info = {}
        camera_info["baseline"] = args.baseline
        camera_info["sensitivity"] = args.sensitivity
        camera_info["gain"] = args.gain
        camera_info["qe"] = args.qe

        if args.fit_method == "mle":
            # use default settings
            convergence = 0.001
            max_iterations = 1000
        else:
            convergence = 0
            max_iterations = 0

        if args.fit_method == "lq-3d" or args.fit_method == "lq-gpu-3d":
            from . import zfit
            print("------------------------------------------")
            print('Fitting 3D')
            magnification_factor = float(input("Enter Magnification factor: "))
            zpath = input("Path to *.yaml calibration file: ")

            if zpath:
                try:
                    with open(zpath, "r") as f:
                        z_calibration = yaml.load(f)
                except Exception as e:
                    print(e)
                    print('Error loading calibration file.')
                    raise

        for i, path in enumerate(paths):
            print("------------------------------------------")
            print("------------------------------------------")
            print("Processing {}, File {} of {}".format(path, i+1, len(paths)))
            print("------------------------------------------")
            movie, info = _io.load_movie(path)
            current, futures = identify_async(movie, min_net_gradient, box)
            n_frames = len(movie)
            while current[0] < n_frames:
                print(
                    "Identifying in frame {:,} of {:,}".format(
                        current[0] + 1, n_frames
                    ),
                    end="\r",
                )
                sleep(0.2)
            print(
                "Identifying in frame {:,} of {:,}".format(n_frames, n_frames)
            )
            ids = identifications_from_futures(futures)

            if args.fit_method == "lq" or args.fit_method == "lq-3d":
                spots = get_spots(movie, ids, box, camera_info)
                theta = gausslq.fit_spots_parallel(spots, asynch=False)
                locs = gausslq.locs_from_fits(ids, theta, box, args.gain)
            elif args.fit_method == "lq-gpu" or args.fit_method == "lq-gpu-3d":
                spots = get_spots(movie, ids, box, camera_info)
                theta = gausslq.fit_spots_gpufit(spots)
                em = camera_info["gain"] > 1
                locs = gausslq.locs_from_fits_gpufit(ids, theta, box, em)
            elif args.fit_method == "mle":
                current, thetas, CRLBs, likelihoods, iterations = fit_async(
                    movie, camera_info, ids, box, convergence, max_iterations
                )
                n_spots = len(ids)
                while current[0] < n_spots:
                    print(
                        "Fitting spot {:,} of {:,}".format(
                            current[0] + 1, n_spots
                        ),
                        end="\r",
                    )
                    sleep(0.2)
                print("Fitting spot {:,} of {:,}".format(n_spots, n_spots))
                locs = locs_from_fits(
                    ids, thetas, CRLBs, likelihoods, iterations, box, config
                )

            elif args.fit_method == "avg":
                spots = get_spots(movie, ids, box, camera_info)
                theta = avgroi.fit_spots_parallel(spots, asynch=False)
                locs = avgroi.locs_from_fits(ids, theta, box, args.gain)

            else:
                print("This should never happen...")

            localize_info = {
                "Generated by": "Picasso Localize",
                "ROI": None,
                "Box Size": box,
                "Min. Net Gradient": min_net_gradient,
                "Convergence Criterion": convergence,
                "Max. Iterations": max_iterations,
            }

            if args.fit_method == "lq-3d" or args.fit_method == "lq-gpu-3d":
                print("------------------------------------------")
                print("Fitting 3D...", end='')
                fs = zfit.fit_z_parallel(locs, info, z_calibration,
                                         magnification_factor,
                                         filter=0, asynch=True)
                locs = zfit.locs_from_futures(fs, filter=0)
                localize_info["Z Calibration Path"] = zpath
                localize_info["Z Calibration"] = z_calibration
                print("complete.")
                print("------------------------------------------")

            info.append(localize_info)

            base, ext = os.path.splitext(path)
            out_path = base + "_locs.hdf5"
            save_locs(out_path, locs, info)
            print("File saved to {}".format(out_path))
            print("drifting : ", args.drift)
            if args.drift > 0:
                print("Undrifting file:")
                print("------------------------------------------")
                try:
                    _undrift(
                        out_path, args.drift, display=False, fromfile=None
                    )
                except Exception as e:
                    print(e)
                    print("Drift correction failed for {}".format(out_path))

            print("                                          ")
            # BEGINING modif
            # Modified by Nicolas Riss to access to locs
            result.append(locs)
            # END modif
    else:
        print("Error. No files found.")
        raise FileNotFoundError
    return result


def launchLocalize(file, config):
    import argparse

    # Main parser
    parser = argparse.ArgumentParser("picasso")


    # localize
    parser.add_argument(
        "files",
        nargs="?",
        default=file,
        help=(
            "one movie file or a folder containing movie files"
            " specified by a unix style path pattern"
        )
    )
    parser.add_argument(
        "-b", "--box-side-length", type=int, default=7, help="box side length"
    )
    parser.add_argument(
        "-a",
        "--fit-method",
        choices=["mle", "lq", "lq-gpu", "lq-3d", "lq-gpu-3d", "avg"],
        default="mle",
    )
    parser.add_argument(
        "-g", "--gradient", type=int, default=int(config['parameters']['localizeGradient']), help="minimum net gradient"
    )
    parser.add_argument(
        "-d",
        "--drift",
        type=int,
        default=1000,
        help="segmentation size for subsequent RCC, 0 to deactivate",
    )
    parser.add_argument(
        "-bl", "--baseline", type=int, default=0, help="camera baseline"
    )
    parser.add_argument(
        "-s", "--sensitivity", type=float, default=1, help="camera sensitivity"
    )
    parser.add_argument(
        "-ga", "--gain", type=int, default=400, help="camera gain" #TODO : test avec 400
    )
    parser.add_argument(
        "-qe", "--qe", type=float, default=1, help="camera quantum efficiency"
    )

    # Parse
    args = parser.parse_args()
    if args.files:
        return _localize(args, config)
    else:
        parser.print_help()
        print("--ERROR--")

if __name__ == "__main__":
    main()
