import csv
import argparse
import os
import numpy as np
import math
import time
import bisect
from .gaussiansmoothing import *
from multiprocessing.pool import Pool



class LocalCalibration:

    def __init__(self, alpha, reverse = None, windowclinvarpoints = 100,
                 windowgnomadfraction = 0.03, gaussian_smoothing = False, pu_smoothing=False):
        self.alpha = alpha
        self.reverse = reverse
        self.windowclinvarpoints = windowclinvarpoints
        self.windowgnomadfraction = windowgnomadfraction
        self.gaussian_smoothing = gaussian_smoothing
        self.pu_smoothing = pu_smoothing


    @staticmethod
    def preprocess_data(x,y,g,reverse):
        if reverse:
            x = np.negative(x)
            g = np.negative(g)
            
        return x,y,g


    @staticmethod
    def findPosterior(allthrs, thrs, xpos, xneg, g, minpoints, gft, w, pu_smoothing):
    
        maxthrs = allthrs[0];
        minthrs = allthrs[-1];
        lengthgnomad = len(g)
        
        smallwindow = 0.0
        bigwindow = maxthrs - minthrs
        
        # check that bigwindow works else we will have to return false
        # if small window works, return smallwindow

        maxiterations = 100
        iterationNo = 0
        pos = None
        neg = None
        eps = 0.000001
        
        conditionSatisfied = False

        while(bigwindow - smallwindow > eps):
            iterationNo += 1
            if iterationNo > 100:
                raise Exception("will have to address this\n")
            currentWindow = (bigwindow + smallwindow)/2.0
            lo = thrs - currentWindow - (eps/2)
            hi = thrs + currentWindow + (eps/2)
            
            c = None
            if hi > maxthrs:
                c = (maxthrs-lo)/(hi-lo)
            elif lo < minthrs:
                c = (hi-minthrs)/(hi-lo)
            else:
                c = 1
            
            if c <= 0:
                raise Exception("Problem with computing c")

            pos = bisect.bisect_right(xpos, hi) - bisect.bisect_left(xpos, lo)
            neg = bisect.bisect_right(xneg, hi) - bisect.bisect_left(xneg, lo)
            if pos + neg < c*minpoints:
                smallwindow = currentWindow
                continue

            if pu_smoothing:
                gfilterlen = bisect.bisect_right(g, hi) - bisect.bisect_left(g, lo)
                if gfilterlen < gft*c*lengthgnomad:
                    smallwindow = currentWindow
                    continue

            bigwindow = currentWindow

        return pos/(pos + w*neg)




    @staticmethod
    def get_both_local_posteriors(x, y, g, thrs, w, minpoints, gft=None, gaussian_smoothing = False, pu_smoothing = False):
        
        xpos = np.sort(x[y==1])  #[x[i] for i in range(len(x)) if y[i] == 1]
        xneg = np.sort(x[y==0])  #[x[i] for i in range(len(x)) if y[i] == 0]
        
        assert x.size == y.size
        gsorted = np.sort(g)
        
        post = np.zeros(len(thrs))
        for i in range(len(thrs)):
            post[i] = LocalCalibration.findPosterior(thrs, thrs[i], xpos, xneg, gsorted, minpoints, gft, w, pu_smoothing)

        if gaussian_smoothing:
            post = gaussian_kernel_smoothing(thrs, post, 5)
        return post
        


    @staticmethod
    def compute_thresholds(l):
        th = np.unique(l).tolist()
        if not math.floor(th[0]) == th[0]:
            th.insert(0, math.floor(th[0]))
        if not math.ceil(th[-1]) == th[-1]:
            th.append(math.ceil(th[-1]))
        th.reverse()
        return th


    def fit(self, X_train, y_train, pu_data, alpha):

        if self.pu_smoothing:
            assert(pu_data is not None)
        # preprocess
        x,y,g = LocalCalibration.preprocess_data(X_train, y_train, pu_data, self.reverse)
        if alpha is None:
            alpha = self.alpha
        w = ( (1-alpha)*((y==1).sum()) ) /  ( alpha*((y==0).sum()) )
        xg = np.concatenate((x,g))
        thresholds = self.compute_thresholds(xg)

        posteriors_p = self.get_both_local_posteriors(x, y, g, thresholds, w, self.windowclinvarpoints, self.windowgnomadfraction, self.gaussian_smoothing, self.pu_smoothing)
        return thresholds, posteriors_p



