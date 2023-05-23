import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import sys
import pickle
import os
#path = os.getcwd()

#path_mm = os.path.dirname(path)
sys.path.append('/home/regnier/work/regnier/MapMaking/')
import qubic
from qubic import NamasterLib as nam
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

N = 40
nstk = 3
nside = 256

path = '/pbs/home/m/mregnier/sps1/mapmaking/frequency_map_making'
seed = np.arange(1, 21, 1)
number_of_seed = len(seed)
iteration = [1, 2]
type = 'band150220_010'
ndet = False
npho150 = True
npho220 = False

def get_file(path, seed, iteration, ndet, npho150, npho220, type):

    path_exp = f'/{type}/'
    
    file = f'MM_maxiter60_convolutionTrue_npointing10000_nrec2_nsub8_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}_iteration{iteration}.pkl'
    #print(path+path_exp+file)

    return path+path_exp+file
def open_file(path):

    ### dict_keys(['maps', 'initial', 'beta', 'gain', 'allfwhm', 'coverage', 'convergence', 'spectra_cmb', 'spectra_dust', 'execution_time'])

    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
def initialize_namaster(pixok, lmin, lmax, dl, nside):

    maskpix = np.zeros(12*nside**2)
    maskpix[pixok] = 1
    aposize=4
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl, aposize=aposize)
    #fsky = maskpix.astype(float).sum() / maskpix.size
    #print(fsky)
    #Namaster.fsky = fsky
    Namaster.ell_binned, _ = Namaster.get_binning(nside)
    return Namaster


lmin = 40
lmax = 2*nside
dl = 35

m_for_nam = np.zeros((2, 12*nside**2, 3))
Dls = np.zeros((len(seed), 13, 4))
### Make loop over iterations

for iseed, s in enumerate(seed):
    for i in iteration:
        print(iseed, s, i)
        filename = get_file(path, s, i, ndet, npho150, npho220, type)
        d = open_file(filename)

        target = np.min(d['allfwhm'])
        C = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(target**2 - np.min(d['allfwhm'])**2))
        seenpix = d['coverage']/d['coverage'].max() > 0.2
        Namaster = initialize_namaster(seenpix, lmin, lmax, dl, nside)
        d['output'][:, ~seenpix] = 0
        m_for_nam[i-1] = C(d['output'][1]).copy()
        
        plt.figure(figsize=(10, 5))
        hp.gnomview(C(d['output'][1])[:, 1], rot=d['center'], reso=15, cmap='jet', min=-8, max=8)
        plt.savefig('testq.png')
        plt.close()
    m_for_nam[:, :, 0] = 0
    leff, Dls[iseed], _ = Namaster.get_spectra(map=m_for_nam[0].T, map2=m_for_nam[1].T, verbose=False, 
                                                beam_correction=np.rad2deg(target), pixwin_correction=False)
    print(Dls[iseed, :, 2])

plt.figure(figsize=(10, 5))
plt.errorbar(leff, np.mean(Dls, axis=0)[:, 2], yerr=np.std(Dls, axis=0)[:, 2], fmt='or', capsize=3)
plt.axhline(0, color='black', ls='--', lw=3)
plt.savefig('DlsBB.png')
plt.close()

rms = []
for i in range(number_of_seed):
    rms += [np.std(Dls[:i], axis=0)[0, 2]]

print(rms)
print()
print()

print('Results :')
print(f'    Average Dl BB : {np.mean(Dls, axis=0)[:, 2]}')
print(f'    Std Dl BB     : {np.std(Dls, axis=0)[:, 2]}')

mydict = {'leff':leff, 'DlsBB':Dls[:, :, 2], 'seenpix':seenpix}
output = open(os.getcwd()+f'/Dls_1_{type}_ndet{ndet}_npho150{npho150}_npho220{npho220}_iteration{number_of_seed}.pkl', 'wb')
pickle.dump(mydict, output)
output.close()







