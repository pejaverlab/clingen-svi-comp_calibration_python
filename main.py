import csv
import os
import numpy as np
import math
import time
import bisect
from Tavtigian.tavtigianutils import get_tavtigian_c, get_tavtigian_thresholds
from configmodule import ConfigModule
from LocalCalibration.gaussiansmoothing import *
from multiprocessing.pool import Pool
from LocalCalibration.LocalCalibration import LocalCalibration
from Tavtigian.Tavtigian import LocalCalibrateThresholdComputation
import time
import matplotlib.pyplot as plt
from utils import *
from infer import *

scoretolabel = {8:"Very Strong Pathogenic", 4:"Strong Pathogenic", 3:"Three Pathogenic", 2:"Moderate Pathogenic",
                1:"Supporting Pathogenic",
                -8:"Very Strong Benign", -4:"Strong Benign", -3:"Three Benign", -2:"Moderate Benign",
                -1:"Supporting Benign", 0:"no evidence"}






def storeResults(outdir, thresholds, posteriors_p, posteriors_b, all_pathogenic, all_benign, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB, Post_p, Post_b):

    fname = os.path.join(outdir, "pathogenic_posterior.txt")
    tosave = np.array([thresholds,posteriors_p]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f', header="score\tposterior")
    fname = os.path.join(outdir,"benign_posterior.txt")
    tosave = np.array([np.flip(thresholds),posteriors_b]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f', header="score\tposterior")

    benign95= [np.percentile(e,5) for e in all_benign]
    pathogenic95 = [np.percentile(e,5) for e in all_pathogenic]

    fname = os.path.join(outdir, "pathogenic95_posterior.txt")
    tosave = np.array([thresholds,pathogenic95]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f', header="score\tposterior")
    fname = os.path.join(outdir,"benign95_posterior.txt")
    tosave = np.array([np.flip(thresholds),benign95]).T
    np.savetxt(fname,tosave , delimiter='\t', fmt='%f', header="score\tposterior")
    

    fig, ax = plt.subplots()
    ax.plot(np.flip(thresholds),posteriors_b , linewidth=2.0)
    ax.plot(np.flip(thresholds),benign95, color='gray', label = "One-sided confidence bound")
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title("Posterior-Score Graph")
    ax.axhline(Post_b[4], linestyle='dotted', color='steelblue', label = "BP4_Supporting : " + str(round(Post_b[4],4)) )
    ax.axhline(Post_b[3], linestyle='dashed', color='steelblue', label = "BP4_Moderate : " + str(round(Post_b[3],4)) )
    ax.axhline(Post_b[2], linestyle='dashdot', color='steelblue', label = "BP4_Moderate+ : " + str(round(Post_b[2],4)) )
    ax.axhline(Post_b[1], linestyle=(5, (10, 3)), color='steelblue', label = "BP4_Strong : " + str(round(Post_b[1],4)) )
    ax.axhline(Post_b[0], linestyle='solid', color='steelblue', label = "BP4_VeryStrong: " + str(round(Post_b[0],4)) ) 
    ax.set_ylim([0.975, 1.001])
    plt.legend(loc="lower left", fontsize="xx-small")
    plt.savefig(os.path.join(outdir,"benign.png"))
    ax.clear()

    ax.plot(thresholds,posteriors_p , linewidth=2.0, color='b')
    ax.plot(thresholds,pathogenic95, color='gray', label = "One-sided confidence bound")
    ax.set_xlabel("score")
    ax.set_ylabel("posterior")
    ax.set_title("Posterior-Score Graph")
    ax.axhline(Post_p[4], linestyle='dotted', color='r' ,label = "PP3_Supporting : " + str(round(Post_p[4],4)))
    ax.axhline(Post_p[3], linestyle='dashed', color='r', label = "PP3_Moderate : " + str(round(Post_p[3],4)))
    ax.axhline(Post_p[2], linestyle='dashdot', color='r', label = "Moderate+ : " + str(round(Post_p[2],4)))
    ax.axhline(Post_p[1], linestyle=(5, (10, 3)), color='r', label = "PP3_Strong : " + str(round(Post_p[1],4)))
    ax.axhline(Post_p[0], linestyle='solid', color='r', label = "PP3_VeryStrong : " + str(round(Post_p[0],4)))
    plt.legend(loc="upper left", fontsize="xx-small")
    plt.savefig(os.path.join(outdir, "pathogenic.png"))

    fname = os.path.join(outdir,"pthresh.txt")
    np.savetxt(fname, pthresh , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,"bthresh.txt")
    np.savetxt(fname, bthresh , delimiter='\t', fmt='%f')

    fname = os.path.join(outdir,"pthreshdiscounted.txt")
    stringtosave = "VeryStrong(+8)\t{:.6f}\nStrong(+4)\t{:.6f}\nThree(+3)\t{:.6f}\nModerate(+2)\t{:.6f}\nSupporting(+1)\t{:.6f}"
    stringtosave = stringtosave.format(DiscountedThresholdP[0],DiscountedThresholdP[1],DiscountedThresholdP[2],DiscountedThresholdP[3],DiscountedThresholdP[4])
    f = open(fname, 'w')
    f.write(stringtosave)
    f.close

    fname = os.path.join(outdir,"bthreshdiscounted.txt")
    stringtosave = "VeryStrong(-8)\t{:.6f}\nStrong(-4)\t{:.6f}\nThree(-3)\t{:.6f}\nModerate(-2)\t{:.6f}\nSupporting(-1)\t{:.6f}"
    stringtosave = stringtosave.format(DiscountedThresholdB[0],DiscountedThresholdB[1],DiscountedThresholdB[2],DiscountedThresholdB[3],DiscountedThresholdB[4])
    f = open(fname, 'w')
    f.write(stringtosave)
    f.close


    
def calibrate(args):
        
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

    x,y = load_labelled_data(labeldatafile)
    g = load_unlabelled_data(udatafile)

    g = np.sort(np.array(g))
    xg = np.concatenate((x,g))

    calib = LocalCalibration(alpha, reverse, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing, data_smoothing)
    thresholds, posteriors_p = calib.fit(x,y,g,alpha)
    posteriors_b = 1 - np.flip(posteriors_p)
    

    calib = LocalCalibrateThresholdComputation(alpha, c, reverse, windowclinvarpoints, windowgnomadfraction, gaussian_smoothing, data_smoothing)
    start = time.time()
    _, posteriors_p_bootstrap = calib.get_both_bootstrapped_posteriors_parallel(x,y, g, B, alpha, thresholds)
    end = time.time()
    print("time: " ,end - start)


    Post_p, Post_b = get_tavtigian_thresholds(c, alpha)


    all_pathogenic = np.row_stack((posteriors_p, posteriors_p_bootstrap))
    all_benign = 1 - np.flip(all_pathogenic, axis = 1)

    pthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_pathogenic, thresholds, Post_p)
    bthresh = LocalCalibrateThresholdComputation.get_all_thresholds(all_benign, np.flip(thresholds), Post_b) 

    DiscountedThresholdP = LocalCalibrateThresholdComputation.get_discounted_thresholds(pthresh, Post_p, B, discountonesided, 'pathogenic')
    DiscountedThresholdB = LocalCalibrateThresholdComputation.get_discounted_thresholds(bthresh, Post_b, B, discountonesided, 'benign')

    storeResults(outdir, thresholds, posteriors_p, posteriors_b, all_pathogenic[1:].T, all_benign[1:].T, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB, Post_p, Post_b)
    
    print("Discounted Thresholds: ", DiscountedThresholdP)


    
def main():

    
    parser = getParser()
    args = parser.parse_args()
    
    if(args.command=="infer"):
        if (args.score):
            ans = infer([args.score], args.calibrated_data_directory)
            print(ans[0],":",scoretolabel[ans[0]])
        elif (args.score_file):
            scores = np.loadtxt(args.score_file)
            ans = infer(scores, args.calibrated_data_directory)
            print(ans)
        return

    if(args.command == "calibrate"):
        calibrate(args)

        
if __name__ == '__main__':
    main()
