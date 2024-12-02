import csv
import argparse
import os
import numpy as np
import math
import time
import bisect
from .tavtigianutils import get_tavtigian_c
from multiprocessing.pool import Pool
from LocalCalibration.LocalCalibration import LocalCalibration



class LocalCalibrateThresholdComputation:


    def __init__(self, alpha,c, reverse = None, windowclinvarpoints = 100, 
                 windowgnomadfraction = 0.03, gaussian_smoothing = True, pu_smoothing=False):

        self.alpha = alpha
        self.c = c
        self.reverse = reverse
        self.windowclinvarpoints = windowclinvarpoints
        self.windowgnomadfraction = windowgnomadfraction
        self.gaussian_smoothing = gaussian_smoothing
        self.pu_smoothing = pu_smoothing


    @staticmethod
    def initialize(x_, y_, g_, w_, thrs_, minpoints_, gft_, B_, gaussian_smoothing_, pu_smoothing_):
        global x
        global y
        global g
        global w
        global thrs
        global minpoints
        global gft
        global B
        global gaussian_smoothing
        global pu_smoothing
        x = x_
        y = y_
        g = g_
        w = w_
        thrs = thrs_
        minpoints  = minpoints_
        gft = gft_
        B = B_
        gaussian_smoothing = gaussian_smoothing_
        pu_smoothing = pu_smoothing_


    @staticmethod
    def get_both_bootstrapped_posteriors(seed):
        np.random.seed(seed)
        qx = np.random.randint(0,len(x),len(x))
        qg = np.random.randint(0,len(g),len(g))
        posteriors_p = LocalCalibration.get_both_local_posteriors(x[qx], y[qx], g[qg], thrs, w, minpoints, gft, gaussian_smoothing, pu_smoothing)
        return posteriors_p
        


    def get_both_bootstrapped_posteriors_parallel(self, x, y, g, B, alpha = None, thresholds = None):

        if alpha is None:
            alpha = self.alpha
        if self.pu_smoothing:
            assert g is not None
        w = ( (1-alpha)*((y==1).sum()) ) /  ( alpha*((y==0).sum()) )

        x,y,g = LocalCalibration.preprocess_data(x, y, g, self.reverse)

        if thresholds is None:
            xg = np.concatenate((x,g))
            thresholds = LocalCalibration.compute_thresholds(xg)

        ans = None
        with Pool(192,initializer = LocalCalibrateThresholdComputation.initialize, initargs=(x, y, g, w, thresholds, self.windowclinvarpoints, self.windowgnomadfraction, B, self.gaussian_smoothing, self.pu_smoothing),) as pool:
            items = [i for i in range(B)]
            ans = pool.map(LocalCalibrateThresholdComputation.get_both_bootstrapped_posteriors, items, 64)

        return thresholds, np.array(ans)



    @staticmethod
    def get_all_thresholds(posteriors, thrs, Post):
    
        thresh = np.zeros((posteriors.shape[0], len(Post)))

        for i in range(posteriors.shape[0]):
            posterior = posteriors[i]
            for j in range(len(Post)):
                idces = np.where(posterior < Post[j])[0]
                if len(idces) > 0 and idces[0] > 0:
                    thresh[i][j] = thrs[idces[0]-1]
                else:
                    thresh[i][j] = np.NaN

        return thresh


    @staticmethod
    def get_discounted_thresholds(thresh, Post, B, discountonesided, tp):
        threshT = thresh.T
        DiscountedThreshold = np.zeros(len(Post))
        for j in range(len(Post)):
            threshCurrent = threshT[j]
            invalids = np.count_nonzero(np.isnan(threshCurrent))
            if invalids > discountonesided*B:
                DiscountedThreshold[j] = np.NaN
            else:
                valids = threshCurrent[~np.isnan(threshCurrent)]
                valids = np.sort(valids)
                if tp == 'pathogenic':
                    valids = np.flip(valids)
                else:
                    assert tp == 'benign'
                    
                DiscountedThreshold[j] = valids[np.floor(discountonesided * B).astype(int) - invalids];

        return DiscountedThreshold


        

    @staticmethod
    def convertProbToPoint(prob, alpha, c):
        
        c0 = c
        c1 = np.sqrt(c0)
        c2 = np.sqrt(c1)
        c3 = np.sqrt(c2)

        
        op = prob * (1-alpha)/ ((1-prob)*alpha)
        opsu = c3

        points = np.log10(op)/np.log10(opsu)

        return points
