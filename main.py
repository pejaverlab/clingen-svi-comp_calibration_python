import csv
import argparse
import os
import numpy as np
import math
import time
import bisect
from Tavtigian.tavtigian import get_tavtigian_c, get_tavtigian_thresholds
from configmodule import ConfigModule
from LocalCalibration.gaussiansmoothing import *
from multiprocessing.pool import Pool
from LocalCalibration.LocalCalibration import LocalCalibration
from Tavtigian.Tavtigian import LocalCalibrateThresholdComputation
import time


def load_labelled_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    x = [float(e[0]) for e in data]
    y = [int(e[1]) for e in data]
    return x,y

def load_unlabelled_data(filepath):
    data = None
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        f.close()
    g = [float(e[0]) for e in data]
    return g


def getParser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--configfile",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tool",
        default="Model",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--outdir",
        default="out",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--labelled_data_file",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--unlabelled_data_file",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--reverse",
        action='store_true',
    )

    return parser

def storeResults(outdir, tool, thresholds, posteriors_p, posteriors_b, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB, Post_p, Post_b):

    fname = os.path.join(outdir,tool + "-pathogenic.txt")
    tosave = np.array([thresholds,posteriors_p]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,tool + "-benign.txt")
    tosave = np.array([np.flip(thresholds),posteriors_b]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(np.flip(thresholds),posteriors_b , linewidth=2.0)
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title(tool)
    ax.axhline(Post_b[4], linestyle='dotted', color='steelblue', label = "BP4_Supporting : " + str(round(Post_b[4],4)) )
    ax.axhline(Post_b[3], linestyle='dashed', color='steelblue', label = "BP4_Moderate : " + str(round(Post_b[3],4)) )
    ax.axhline(Post_b[2], linestyle='dashdot', color='steelblue', label = "BP4_Moderate+ : " + str(round(Post_b[2],4)) )
    ax.axhline(Post_b[1], linestyle=(5, (10, 3)), color='steelblue', label = "BP4_Strong : " + str(round(Post_b[1],4)) )
    ax.axhline(Post_b[0], linestyle='solid', color='steelblue', label = "BP4_VeryStrong: " + str(round(Post_b[0],4)) ) 
    ax.set_ylim([0.975, 1.001])
    plt.legend()
    plt.savefig(os.path.join(outdir,tool+"-benign.png"))
    ax.clear()

    ax.plot(thresholds,posteriors_p , linewidth=2.0, color='b')
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title(tool)
    ax.axhline(Post_p[4], linestyle='dotted', color='r' ,label = "PP3_Supporting : " + str(round(Post_p[4],4)))
    ax.axhline(Post_p[3], linestyle='dashed', color='r', label = "PP3_Moderate : " + str(round(Post_p[3],4)))
    ax.axhline(Post_p[2], linestyle='dashdot', color='r', label = "Moderate+ : " + str(round(Post_p[2],4)))
    ax.axhline(Post_p[1], linestyle=(5, (10, 3)), color='r', label = "PP3_Strong : " + str(round(Post_p[1],4)))
    ax.axhline(Post_p[0], linestyle='solid', color='r', label = "PP3_VeryStrong : " + str(round(Post_p[0],4)))
    plt.legend()
    plt.savefig(os.path.join(outdir, tool+"-pathogenic.png"))

    fname = os.path.join(outdir,tool + "-pthresh.txt")
    np.savetxt(fname, pthresh , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,tool + "-bthresh.txt")
    np.savetxt(fname, bthresh , delimiter='\t', fmt='%f')

    fname = os.path.join(outdir,tool + "-pthreshdiscounted.txt")
    np.savetxt(fname, DiscountedThresholdP , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,tool + "-bthreshdiscounted.txt")
    np.savetxt(fname, DiscountedThresholdB , delimiter='\t', fmt='%f')


    

def main():

    parser = getParser()
    args = parser.parse_args()
    tool = args.tool
    labeldatafile = args.labelled_data_file
    udatafile = args.unlabelled_data_file
    reverse = args.reverse
    configfile = args.configfile
    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    configmodule = ConfigModule()
    configmodule.load_config(configfile)
    B = configmodule.B
    discountonesided = configmodule.discountonesided
    windowclinvarpoints = configmodule.windowclinvarpoints
    windowgnomadfraction = configmodule.windowgnomadfraction
    gaussian_smoothing = configmodule.gaussian_smoothing
    data_smoothing = configmodule.data_smoothing
    if data_smoothing:
        assert udatafile is not None

    alpha = None
    c = None
    if (configmodule.emulate_tavtigian):
        alpha = 0.1
        c = 350
    elif (configmodule.emulate_pejaver):
        alpha = 0.0441
        c = 1124
    else:
        alpha = configmodule.alpha
        c = get_tavtigian_c(alpha)

    print(c)



    x,y = load_labelled_data(labeldatafile)
    g = load_unlabelled_data(udatafile)
    print(g[:100])
    #return

    x = np.array(x)
    y = np.array(y)
    g = np.sort(np.array(g))
    xg = np.concatenate((x,g))

    calib = LocalCalibration(alpha, c, reverse, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing, data_smoothing)
    thresholds, posteriors_p = calib.fit(x,y,g,alpha)
    posteriors_b = 1 - np.flip(posteriors_p)
    

    calib = LocalCalibrateThresholdComputation(alpha, c, reverse, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing, data_smoothing)
    start = time.time()
    _, posteriors_p_bootstrap = calib.get_both_bootstrapped_posteriors_parallel(x,y, g, 1000, alpha, thresholds)
    end = time.time()
    print("time: " ,end - start)


    Post_p, Post_b = get_tavtigian_thresholds(c, alpha)

    print("Post P and B")
    print(Post_p)
    print(Post_b)

    all_pathogenic = np.row_stack((posteriors_p, posteriors_p_bootstrap))
    all_benign = 1 - np.flip(all_pathogenic, axis = 1)

    pthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_pathogenic, thresholds, Post_p)
    bthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_benign, np.flip(thresholds), Post_b) 

    DiscountedThresholdP = LocalCalibrateThresholdComputation.get_discounted_thresholds(pthresh, Post_p, B, discountonesided, 'pathogenic')
    DiscountedThresholdB = LocalCalibrateThresholdComputation.get_discounted_thresholds(bthresh, Post_b, B, discountonesided, 'benign')

    storeResults(outdir, tool, thresholds, posteriors_p, posteriors_b, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB, Post_p, Post_b)
    
    print("Thresholds: ", pthresh)
    print("Discounted Thresholds: ", DiscountedThresholdP)



if __name__ == '__main__':
    main()
