from scipy.optimize import fsolve
import numpy as np

def evidence_to_plr(opvst, npsu, npm, npst, npvst):
    return opvst**( (npsu/8) + (npm/4) + (npst/2) + npvst )

def odds_to_postP(plr, priorP):
    postP = plr * priorP / ((plr - 1) * priorP + 1)
    return postP

def get_postP(c, prior, npsu, npm, npst, npvst):
    plr = evidence_to_plr(c, npsu, npm, npst, npvst)
    return odds_to_postP(plr, prior)

def get_postP_moderate(c, prior):
    return get_postP(c,prior,2,0,1,0) - 0.9

def get_tavtigian_c(prior):
    return fsolve(get_postP_moderate, 300 , args=(prior))

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

    #for j in range(4):                                                                                                                                                                                                                 
    #    Post_p[j] = c ** (1 / 2 ** (j)) * alpha / ((c ** (1 / 2 ** (j)) - 1) * alpha + 1);                                                                                                                                             
    #    Post_b[j] = (c ** (1 / 2 ** (j))) * (1 - alpha) /(((c ** (1 / 2 ** (j))) - 1) * (1 - alpha) + 1);                                                                                                                              

    return Post_p, Post_b

