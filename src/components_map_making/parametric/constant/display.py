import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
import os
import qubic
path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(path)

from analysis import AnalysisParametricConstant

center = qubic.equ2gal(100, -157)
filename = 'Two_ndet0_pho1500_pho2200_seed1_iteration1'

N = 30
duration = 50
nc = 2
nside = 128
#make_gif(self, icomp, istk, rot, reso, min=-4, max=4, minr=-2, maxr=2)

a = AnalysisParametricConstant(filename, nside, N, nc)
a.plot_all_components(-1, center, reso=15, istk=1, name=['CMB', 'Dust @ 150 GHz', r'CO - $J_{1 2}$'], fwhm=np.sqrt(0.0078**2 - 0.0044**2))
a.plot_convergence_maps()
#a.make_gif_gnomview(duration=duration, icomp=0, istk=1, rot=center, reso=15, min=-6, max=6, minr=-6, maxr=6)
#a.make_gif_gnomview(duration=duration, icomp=1, istk=1, rot=center, reso=15, min=-15, max=15, minr=-6, maxr=6)
#a.make_gif_gnomview(duration=duration, icomp=2, istk=1, rot=center, reso=15, min=-2, max=5, minr=-2, maxr=2)
#a.make_gif_gnomview(duration=duration, icomp=2, istk=0, rot=center, reso=15, min=0, max=100, minr=-8, maxr=8)
#a.make_gif_beta(duration=duration, truth=1.54, alpha=0.5, log=True)