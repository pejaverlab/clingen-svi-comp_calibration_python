from scipy.optimize import fsolve
import numpy as np

def evidence_to_plr(opvst, npsu=2, npm=0, npst=1, npvst=0):
    return opvst**( (npsu/8) + (npm/4) + (npst/2) + npvst )

def odds_to_postP(plr, priorP):
    postP = plr * priorP / ((plr - 1) * priorP + 1)
    return postP

def get_postP(c, prior, npsu=2, npm=0, npst=1, npvst=0):
    plr = evidence_to_plr(c, npsu, npm, npst, npvst)
    return odds_to_postP(plr, prior)

def get_postP_moderate(c, prior):
    return get_postP(c,prior,2,0,1,0) - 0.9

#def get_plr_moderate(c):
#    return evidence_to_plr(opvst, 2, 0, 1, 0)

def get_tavtigian_c(prior):
    return fsolve(get_postP_moderate, 300 , args=(prior))

def get_tavtigian_plr(c):

    plr = np.zeros(5, dtype=int)

    plr[0] = int(c ** (8 / 8));
    plr[1] = int(c ** (4 / 8));
    plr[2] = int(c ** (3 / 8));
    plr[3] = int(c ** (2 / 8));
    plr[4] = int(c ** (1 / 8)); 

    return plr


def get_tavtigian_thresholds(c, alpha):

    Post_p = np.zeros(5)
    Post_b = np.zeros(5)

    Post_p[0] = c ** (8 / 8) * alpha / ((c ** (8 / 8) - 1) * alpha + 1);
    Post_p[1] = c ** (4 / 8) * alpha / ((c ** (4 / 8) - 1) * alpha + 1);
    Post_p[2] = c ** (3 / 8) * alpha / ((c ** (3 / 8) - 1) * alpha + 1);
    Post_p[3] = c ** (2 / 8) * alpha / ((c ** (2 / 8) - 1) * alpha + 1);
    Post_p[4] = c ** (1 / 8) * alpha / ((c ** (1 / 8) - 1) * alpha + 1);

    Post_b[0] = (c ** (8 / 8)) * (1 - alpha) /(((c ** (8 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[1] = (c ** (4 / 8)) * (1 - alpha) /(((c ** (4 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[2] = (c ** (3 / 8)) * (1 - alpha) /(((c ** (3 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[3] = (c ** (2 / 8)) * (1 - alpha) /(((c ** (2 / 8)) - 1) * (1 - alpha) + 1);
    Post_b[4] = (c ** (1 / 8)) * (1 - alpha) /(((c ** (1 / 8)) - 1) * (1 - alpha) + 1);

    return Post_p, Post_b

