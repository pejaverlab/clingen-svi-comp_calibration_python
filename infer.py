import numpy as np
import os

def infer_single(score, pthreshdiscounted, bthreshdiscounted):
    if(not np.isnan(pthreshdiscounted[0]) and score > pthreshdiscounted[0]):
        return 8
    if(not np.isnan(pthreshdiscounted[1]) and score > pthreshdiscounted[1]):
        return 4
    if(not np.isnan(pthreshdiscounted[2]) and score > pthreshdiscounted[2]):
        return 3
    if(not np.isnan(pthreshdiscounted[3]) and score > pthreshdiscounted[3]):
        return 2
    if(not np.isnan(pthreshdiscounted[4]) and score > pthreshdiscounted[4]):
        return 1
    if(not np.isnan(bthreshdiscounted[0]) and score < bthreshdiscounted[0]):
        return -8
    if(not np.isnan(bthreshdiscounted[1]) and score < bthreshdiscounted[1]):
        return -4
    if(not np.isnan(bthreshdiscounted[2]) and score < bthreshdiscounted[2]):
        return -3
    if(not np.isnan(bthreshdiscounted[3]) and score < bthreshdiscounted[3]):
        return -2
    if(not np.isnan(bthreshdiscounted[4]) and score < bthreshdiscounted[4]):
        return -1
    else:
        return 0

def infer_single_reverse(score, pthreshdiscounted, bthreshdiscounted):
    if(not np.isnan(pthreshdiscounted[0]) and score < pthreshdiscounted[0]):
        return 8
    if(not np.isnan(pthreshdiscounted[1]) and score < pthreshdiscounted[1]):
        return 4
    if(not np.isnan(pthreshdiscounted[2]) and score < pthreshdiscounted[2]):
        return 3
    if(not np.isnan(pthreshdiscounted[3]) and score < pthreshdiscounted[3]):
        return 2
    if(not np.isnan(pthreshdiscounted[4]) and score < pthreshdiscounted[4]):
        return 1
    if(not np.isnan(bthreshdiscounted[0]) and score > bthreshdiscounted[0]):
        return -8
    if(not np.isnan(bthreshdiscounted[1]) and score > bthreshdiscounted[1]):
        return -4
    if(not np.isnan(bthreshdiscounted[2]) and score > bthreshdiscounted[2]):
        return -3
    if(not np.isnan(bthreshdiscounted[3]) and score > bthreshdiscounted[3]):
        return -2
    if(not np.isnan(bthreshdiscounted[4]) and score > bthreshdiscounted[4]):
        return -1
    else:
        return 0


def readDiscoutedThresholdFile(filename):
    thresholds95 = []
    plr95 = []
    ftext = open(filename, "r").read().split("\n")[1:]
    for l in ftext:
        v = l.split('\t')
        thresholds95.append(float(v[1]))
        plr95.append(int(v[2]))
        #thresholds95 = [float(e.split('\t')[1]) for e in ftext]
    return thresholds95, plr95
    
        
    
def infer_evidence(scores, pthreshdiscounted, bthreshdiscounted, reverse=False):
    if(not reverse):
        return [infer_single(e,pthreshdiscounted, bthreshdiscounted) for e in scores]
    else:
        return [infer_single_reverse(e,pthreshdiscounted, bthreshdiscounted) for e in scores]

