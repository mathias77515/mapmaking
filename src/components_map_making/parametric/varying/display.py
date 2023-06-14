import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os
import qubic
path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(path)

from analysis import AnalysisParametricVarying

center = qubic.equ2gal(0, -57)
filename = 'x0cmb0_Two_ndetFalse_pho150False_pho220False_seed1_iteration1'

N = 100
duration = 100
nside = 128
nside_fit = 8

#make_gif(self, icomp, istk, rot, reso, min=-4, max=4, minr=-2, maxr=2)
a = AnalysisParametricVarying(filename, nside, nside_fit, N, 2)
#a.make_gif_gnomview(icomp=0, istk=1, duration=duration,
#                    rot=center, reso=25, min=-4, max=4, minr=-0.0001, maxr=0.0001)

#a.make_gif_gnomview(icomp=1, istk=1, duration=duration,
#                    rot=center, reso=25, min=-12, max=12, minr=-0.0001, maxr=0.0001)
#a.plot_beta(bar_ite=1, truth=1.54, alpha=0.3)
#a.make_gif_gnomview(duration=duration, icomp=0, istk=1, rot=center, reso=15, min=-6, max=6, minr=-6, maxr=6)
#a.make_gif_beta(duration=duration, truth=1.54, alpha=0.5)