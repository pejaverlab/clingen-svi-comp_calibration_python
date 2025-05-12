import csv
import os
import numpy as np
import math
import time
import bisect
from Tavtigian.tavtigianutils import get_tavtigian_c, get_tavtigian_thresholds, get_tavtigian_plr
from configmodule import ConfigModule
from LocalCalibration.gaussiansmoothing import *
from multiprocessing.pool import Pool
from LocalCalibration.LocalCalibration import LocalCalibration
from Tavtigian.Tavtigian import LocalCalibrateThresholdComputation
import time
import matplotlib.pyplot as plt
from utils import *
from infer import *
import datetime
import json

scoretolabel = {8:"PP3_VeryStrong", 4:"PP3_Strong", 3:"PP3_+3", 2:"PP3_Moderate",
                1:"PP3_Supporting",
                -8:"BP4_VeryStrong", -4:"BP4_Strong", -3:"BP4_-3", -2:"BP4_Moderate",
                -1:"BP4_Supporting", 0:"Variant of Uncertain Significance"}

pthreshdiscountedvalues = {"BayesDel-noAF":   [np.nan, 0.500, 0.410, 0.270, 0.130],
                           "MutPred2.0":      [np.nan, 0.932, 0.895, 0.829, 0.737],
                           "REVEL":   [np.nan, 0.932,   0.879, 0.773, 0.644],
                           "VEST4":   [np.nan, 0.965, 0.909, 0.861, 0.764],
                           "AlphaMissense":   [np.nan, 0.990, 0.973, 0.906, 0.792],
                           "ESM1b":   [np.nan, -24.000, -14.028, -12.253, -10.651],
                           "VARITY-R":        [np.nan, 0.965, 0.915, 0.842, 0.675]
                           }

bthreshdiscountedvalues = {"BayesDel-noAF":   [np.nan, np.nan, -0.520,  -0.360,  -0.180],
                           "MutPred2.0":      [np.nan, 0.010, 0.031, 0.197, 0.391],
                           "REVEL":   [0.003, 0.016, 0.052, 0.183, 0.290],
                           "VEST4":   [np.nan, np.nan, 0.077, 0.302, 0.449],
                           "AlphaMissense":   [np.nan, np.nan, 0.070, 0.099, 0.169],
                           "ESM1b":   [np.nan, np.nan,  8.841,   -3.098,  -6.268],
                           "VARITY-R": [np.nan, 0.036, 0.063, 0.116, 0.251]
                           }

tool_direction_reverse = {"BayesDel-noAF": False,
                          "MutPred2.0": False,
                          "REVEL": False,
                          "VEST4": False,
                          "AlphaMissense": False,
                          "ESM1b": True,
                          "VARITY-R": False
                          }


def storeResults(outdir, thresholds, posteriors_p, posteriors_b, all_pathogenic, all_benign, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB, Post_p, Post_b, plr):

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
    ax.set_ylim([max(0,Post_b[4]-0.2*(Post_b[0]-Post_b[4])), 1.001])
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
    ax.set_ylim([max(0,Post_p[4]-0.2*(Post_p[0]-Post_p[4])), 1.001])
    plt.legend(loc="upper left", fontsize="xx-small")
    plt.savefig(os.path.join(outdir, "pathogenic.png"))

    fname = os.path.join(outdir,"pthresh.txt")
    np.savetxt(fname, pthresh , delimiter='\t', fmt='%f')
    fname = os.path.join(outdir,"bthresh.txt")
    np.savetxt(fname, bthresh , delimiter='\t', fmt='%f')

    print("Discounted Thresholds Saving: ", DiscountedThresholdP)
    print(plr)
    fname = os.path.join(outdir,"pthreshdiscounted.txt")
    stringtosave = "AMCG\tScoreThreshold\tOddsOfPathogenicity\n"
    stringtosave += "VeryStrong(+8)\t{:.6f}\t{}\nStrong(+4)\t{:.6f}\t{}\nThree(+3)\t{:.6f}\t{}\nModerate(+2)\t{:.6f}\t{}\nSupporting(+1)\t{:.6f}\t{}"
    stringtosave = stringtosave.format(DiscountedThresholdP[0],plr[0],DiscountedThresholdP[1],plr[1],DiscountedThresholdP[2],plr[2],DiscountedThresholdP[3],plr[3],DiscountedThresholdP[4],plr[4])
    f = open(fname, 'w')
    f.write(stringtosave)
    f.close

    fname = os.path.join(outdir,"bthreshdiscounted.txt")
    stringtosave = "AMCG\tScoreThreshold\tOddsOfBenignity\n"
    stringtosave += "VeryStrong(-8)\t{:.6f}\t{}\nStrong(-4)\t{:.6f}\t{}\nThree(-3)\t{:.6f}\t{}\nModerate(-2)\t{:.6f}\t{}\nSupporting(-1)\t{:.6f}\t{}"
    stringtosave = stringtosave.format(DiscountedThresholdB[0],plr[0],DiscountedThresholdB[1],plr[1],DiscountedThresholdB[2],plr[2],DiscountedThresholdB[3],plr[3],DiscountedThresholdB[4],plr[4])
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

    plr = get_tavtigian_plr(c)

    storeResults(outdir, thresholds, posteriors_p, posteriors_b, all_pathogenic[1:].T, all_benign[1:].T, pthresh, bthresh, DiscountedThresholdP, DiscountedThresholdB, Post_p, Post_b, plr)
    
    print("Discounted Thresholds: ", DiscountedThresholdP)

    configmodule.save_config(outdir)


def infer(args):
    
    if(args.calibrated_data_directory is not None):
        pthreshdiscounted, pplr = readDiscoutedThresholdFile(os.path.join(args.calibrated_data_directory,"pthreshdiscounted.txt"))
        bthreshdiscounted, bplr = readDiscoutedThresholdFile(os.path.join(args.calibrated_data_directory,"bthreshdiscounted.txt"))
        reverse = args.reverse
    elif(args.tool_name):
        print("Prior 4.41%")
        pthreshdiscounted = pthreshdiscountedvalues[args.tool_name]
        bthreshdiscounted = bthreshdiscountedvalues[args.tool_name]
        pplr = [1124,33,13,5,2]
        bplr = [1124,33,13,5,2]
        reverse = tool_direction_reverse[args.tool_name]

    
    if (args.score):
        ans = infer_evidence([args.score], pthreshdiscounted, bthreshdiscounted, reverse)
        print(datetime.datetime.now().strftime("#%I:%M%p on %B %d, %Y\n"))
        print("Odds of Pathogenicity/Benignity\n"+ "Very Strong(8): " + str(pplr[0]) + "\nPLR Strong(4): " + str(pplr[1]) +
              "\nPLR +3(3): " + str(pplr[2])+
              "\nPLR Moderate(2): " + str(pplr[3]) + "\nPLR Supporting(1): " + str(pplr[4]) + "\n"  )
        print("Score:\t" + str(args.score) + "\nACMG20:\t" + str(ans[0]) + "\nACMG18: " +  scoretolabel[ans[0]] + "\n")
    elif (args.score_file):
        scores = np.loadtxt(args.score_file)
        ans = infer_evidence(scores, pthreshdiscounted, bthreshdiscounted, reverse)
        with open("infer_out.txt", "w") as f:
            f.write(datetime.datetime.now().strftime("#%I:%M%p on %B %d, %Y\n"))
            if(args.tool_name):
                f.write("#Method: " + args.tool_name + "\n")
                f.write("#Prior:  4.41%\n")
            elif(args.calibrated_data_directory):
                f.write("#Calibration Scores Obtained From: " + args.calibrated_data_directory + "\n")
            f.write("#PP3: Pathogenic\n#BP4: Benign\n")
            f.write("\nOdds of Pathogenicity/Benignity\n")
            f.write("PLR Very Strong(8): " + str(pplr[0]) + "\nPLR Strong(4): " + str(pplr[1]) + "\nPLR +3(3): " + str(pplr[2])+
                    "\nPLR Moderate(2): " + str(pplr[3]) + "\nPLR Supporting(1): " + str(pplr[4]) + "\n\n"  )
            f.write("Score\tACMG20\tACMG18\n")
            for i in range(len(scores)):
                f.write(str(scores[i]) + "\t" + str(ans[i]) + "\t" + scoretolabel[ans[i]] + "\n")
        f.close()
        print("Evidence stored in infer_out.txt")
        
    
def main():

    
    parser = getParser()
    args = parser.parse_args()
    
    if(args.command == "calibrate"):
        calibrate(args)

    if(args.command=="infer"):
        infer(args)
        
    return


        
if __name__ == '__main__':
    main()
