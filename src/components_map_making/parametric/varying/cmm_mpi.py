# QUBIC packages
import qubic
import sys
import os

if os.environ.get('OMP_NUM_THREADS') is None:
    omp_num_threads = os.cpu_count()
else:
    omp_num_threads = int(os.environ.get('OMP_NUM_THREADS'))

path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

import component_acquisition as Acq
import pickle
import gc
import pysm3

# Display packages
import healpy as hp
import matplotlib.pyplot as plt

# FG-Buster packages
import component_model as c
import mixing_matrix as mm

# General packages
import numpy as np
import warnings
import solver4mpi

from scipy.optimize import minimize
from functools import partial
import time
import configparser
from noise_timeline import QubicNoise, QubicWideBandNoise, QubicDualBandNoise
from planck_timeline import ExternalData2Timeline

# PyOperators packages
from pyoperators import *
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator
from cg import pcg

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f'You requested for {size} processes, this one is the number {rank}')
print(f'You are asking for {omp_num_threads} CPUs per tasks.')

warnings.filterwarnings("ignore")
path = '/home/regnier/work/regnier/MapMaking/ComponentMapMaking/forecast_wideband'

seed = int(sys.argv[1])
iteration = int(sys.argv[2])
ndet = int(sys.argv[3])
npho150 = int(sys.argv[4])
npho220 = int(sys.argv[5])

### Reading and loading configuration file
def load_config(config_file):
    # Créer un objet ConfigParser
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    # Itérer sur chaque section et option
    external = []
    allnus = [30, 44, 70, 100, 143, 217, 353]
    k = 0
    for section in config.sections():
        for option in config.options(section):
            
            # Récupérer la valeur de chaque option de configuration
            value = config.get(section, option)
                
            if section == 'EXTERNAL DATA':
                if value.lower() == 'true':
                    external.append(allnus[k])
                
                k+=1

            # Convertir la valeur en liste si elle est de la forme "1, 2, 3"
            if ',' in value:
                value = [x.strip() for x in value.split(',')]

            # Convertir la valeur en int, float ou bool si c'est possible
            elif value.isdigit():
                value = int(value)
            elif '.' in value and all(part.isdigit() for part in value.split('.')):
                value = float(value)
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'

            # Définir chaque option de configuration en tant que variable globale
            globals()[option] = value
            
    return external
def get_ultrawideband_config():
    
    nu_up = 247.5
    nu_down = 131.25
    nu_ave = np.mean(np.array([nu_up, nu_down]))
    delta = nu_up - nu_ave
    
    return nu_ave, 2*delta/nu_ave
def get_dict(args={}):
    
    '''
    Function for modify the qubic dictionary.
    '''
    ### Get the default dictionary
    dictfilename = 'dicts/pipeline_demo.dict'
    d = qubic.qubicdict.qubicDict()
    d.read_from_file(dictfilename)
    for i in args.keys():
        
        d[str(i)] = args[i]
    
    return d
def give_me_intercal(D, d):
    return 1/np.sum(D[:]**2, axis=1) * np.sum(D[:] * d[:], axis=1)

nu_ave, delta_nu_over_nu = get_ultrawideband_config()

#########################################################################################################
############################################## Arguments ################################################
#########################################################################################################

external = load_config('config.ini')

if nside_fit == 0 :
    raise TypeError('You must have to put nside_fit > 0 (constant spectral index)')

if rank == 0:
    print('************ Configuration of the simulation ************\n')
    print('Instrument          :')
    print(f'    Type           : {type}')
    print(f'    Nsub           : {nsub}')
    print(f'    Pointings      : {npointings}')
    print('Pixelization        :')
    print(f'    Nside          : {nside}\n')
    print('Foregrounds         :')
    print(f'    Seed           : {seed}')
    print(f'    Iteration      : {seed}')
    print(f'Parallelization    : ')
    print(f'    # of MPI tasks : {size}')
    print(f'    # of CPUs      : {omp_num_threads}')



### Create repository to save all pickle files
if prefix is None:
    prefix=''
save_each_ite = f'{prefix}_{type}_ndet{ndet}_pho150{npho150}_pho220{npho220}_seed{seed}_iteration{iteration}'
if rank == 0:
    pass
    #os.makedirs(save_each_ite)
path_to_save = str(save_each_ite)

#########################################################################################################
############################################## Dictionnary ##############################################
#########################################################################################################

comp = []
comp_name = []
nb_param = 0

### Add each components to comp list for mixing matrix computation
if cmb :
    comp.append(c.CMB())
    comp_name.append('CMB')
if dust[0].lower() == 'true':
    comp.append(c.Dust(nu0=nu0_d, temp=temp))
    comp_name.append('DUST')
    nb_param += 1
if synchrotron[0].lower() == 'true':
    comp.append(c.Synchrotron(nu0=nu0_s))                     # We remove a template of synchrotron emission -> fixing the spectral index
    comp_name.append('SYNCHROTRON')
    nb_param += 1
if coline[0].lower() == 'true':
    comp.append(c.COLine(nu=float(coline[2])/1e9, active=False))
    comp_name.append('CO')

A = mm.MixingMatrix(*comp)

### Compute the place where the components are
if cmb :
    i_cmb = A.components.index('CMB')
if dust[0].lower() == 'true':
    i_d = A.components.index('Dust')
if synchrotron[0].lower() == 'true':
    i_sync = A.components.index('Synchrotron')


if size == 1:
    comm = None

### Dictionary for reconstruction
d = get_dict({'npointings':npointings, 'nf_recon':1, 'nf_sub':nsub, 'nside':nside, 'MultiBand':True, 'period':1,
              'filter_nu':nu_ave*1e9, 'noiseless':False, 'comm':comm, 'nprocs_sampling':1, 'nprocs_instrument':size,
              'photon_noise':True, 'nhwp_angles':3, 'effective_duration':3, 'filter_relative_bandwidth':delta_nu_over_nu, 
              'type_instrument':'wide', 'TemperatureAtmosphere150':None, 'TemperatureAtmosphere220':None,
              'EmissivityAtmosphere150':None, 'EmissivityAtmosphere220':None, 'RA_center':100, 'DEC_center':-157})

center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
#########################################################################################################
############################################## Acquisitions #############################################
#########################################################################################################

# QUBIC Acquisition
myqubic = Acq.QubicFullBandComponentsMapMakingParametric(d, Nsub=nsub, comp=comp, kind=type)

# Add external data
allexp = Acq.QubicOtherIntegratedComponentsMapMakingParametric(myqubic, external, comp=comp, nintegr=nintegr)
others = Acq.OtherDataParametric(external, nside, comp=comp, nintegr=nintegr)

### See if we want to fit CO line
isco = coline[0].lower() == 'true'
if isco == False:
    nu_co = None
else:
    nu_co = float(coline[2])

coverage = myqubic.get_coverage()
pixok = coverage[0]/coverage[0].max() > thr
pixok_nside_fit = hp.ud_grade(pixok, nside_fit)
index_seenpix = np.where(pixok_nside_fit == True)[0]
pixok1 = coverage/coverage.max() > thr_cut_planck
if rank == 0:
    print(index_seenpix)
    print(f"We'll fit {len(index_seenpix)} beta")



# Input beta
beta = np.ones((12*nside_pix**2, nb_param))

if str(dust[1]) == 'd0':

    if dust[0].lower() == 'true':
        beta[:, i_d-1] *= 1.54
    if synchrotron[0].lower() == 'true':
        beta[:, i_sync-1] *= -3
elif str(dust[1]) == 'd1':
    if dust[0].lower() == 'true':
        beta[:, i_d-1] = np.array(pysm3.Sky(nside=nside_pix, preset_strings=['d1']).components[0].mbb_index)
    if synchrotron[0].lower() == 'true':
        beta[:, i_sync-1] = np.array(pysm3.Sky(nside=nside_pix, preset_strings=['s1']).components[0].pl_index)
else:
    raise TypeError('not implemented yet..')

#########################################################################################################
############################################## Components ###############################################
#########################################################################################################

dcomp = {}
if cmb:
    dcomp['cmb'] = seed
if dust[0].lower() == 'true':
    dcomp['dust'] = 'd0'
    beta_d_mean = 1.54
if synchrotron[0].lower() == 'true':
    dcomp['synchrotron'] = 's0'
    beta_s_mean = -3
if coline[0].lower() == 'true':
    dcomp['coline'] = 'co2'

components = myqubic.get_PySM_maps(dcomp).T

# invN
mask = np.ones(12*nside**2)*weigth_reduce_planck
mask[~pixok1[0]] = 1
invN = allexp.get_invntt_operator(mask=mask)
M = Acq.get_preconditioner(np.ones(12*allexp.nside**2))

#########################################################################################################
############################################## Reconstruction ###########################################
#########################################################################################################

if convolution:
    myfwhm = np.sqrt(myqubic.allfwhm**2 - np.min(myqubic.allfwhm)**2)
else:
    myfwhm = None

print(f'FWHM for Nsub : {myfwhm}')

# Get reconstruction operator
H = allexp.get_operator(beta, convolution, co=nu_co)

array_of_operators = myqubic.operator
array_of_operators150 = myqubic.operator[:nsub]
array_of_operators220 = myqubic.operator[nsub:2*nsub]

Hrecon = allexp.get_operator(beta, convolution, list_fwhm=myfwhm, co=nu_co)

# Get simulated data
tod = H(components)

#########################################################################################################
############################################## Systematics ##############################################
#########################################################################################################

_r = ReshapeOperator(myqubic.Ndets*myqubic.Nsamples, (myqubic.Ndets, myqubic.Nsamples))
np.random.seed(None)
#if type == 'Wide':
#    g = np.random.randn(myqubic.Ndets) * sig_gain + 1
#    g /= g[0]
#    G = DiagonalOperator(g, broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
#    tod[:(myqubic.Ndets*myqubic.Nsamples)] = _r.T(G * _r(tod[:(myqubic.Ndets*myqubic.Nsamples)]))
#    print('Gain : ', g[:5])
#elif type == 'Two':
#    g150 = np.random.randn(myqubic.Ndets) * sig_gain + 1
#    g220 = np.random.randn(myqubic.Ndets) * sig_gain + 1
#    g150 /= g150[0]
#    g220 /= g220[0]
#    G150 = DiagonalOperator(g150, broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
#    G220 = DiagonalOperator(g220, broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
#    tod[:(myqubic.Ndets*myqubic.Nsamples)] = _r.T(G150 * _r(tod[:(myqubic.Ndets*myqubic.Nsamples)]))
#    tod[(myqubic.Ndets*myqubic.Nsamples):(2*myqubic.Ndets*myqubic.Nsamples)] = _r.T(G220 * _r(tod[(myqubic.Ndets*myqubic.Nsamples):(2*myqubic.Ndets*myqubic.Nsamples)]))
#    print('Gain 150 : ', g150[:5])
#    print('Gain 220 : ', g220[:5])

if type == 'Wide':
    nq = QubicWideBandNoise(d, npointings).total_noise(int(ndet), int(npho150), int(npho220)).ravel()
elif type == 'Two':
    nq = QubicDualBandNoise(d, npointings).total_noise(int(ndet), int(npho150), int(npho220)).ravel()

seed_pl = 42
n = others.get_noise(seed=seed_pl) * level_noise_planck

n = np.r_[nq, n]

tod += n.copy()

if convolution:
    tod = allexp.reconvolve_to_worst_resolution(tod)

### Separe noisy TOD of QUBIC with external data
if type == 'Two':
    tod_150 = tod[:(myqubic.Ndets*myqubic.Nsamples)]
    tod_220 = tod[(myqubic.Ndets*myqubic.Nsamples):(myqubic.Ndets*myqubic.Nsamples*2)]
    tod_external = tod[((myqubic.Ndets*myqubic.Nsamples)*2):]
elif type == 'Wide':
    tod_w = tod[:(myqubic.Ndets*myqubic.Nsamples)]
    tod_external = tod[((myqubic.Ndets*myqubic.Nsamples)):]

### Define Convolution operator with true kernel
if convolution:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=myqubic.allfwhm[-1], lmax=2*nside-1)
else:
    Ctrue = HealpixConvolutionGaussianOperator(fwhm=0.0, lmax=2*nside-1)

target = np.sqrt(myqubic.allfwhm[0]**2 - myqubic.allfwhm[-1]**2)
C_target = HealpixConvolutionGaussianOperator(fwhm=target)
comp_for_pcg = components.copy()

for i in range(len(comp)):
    if comp_name[i] == 'CMB':
        comp_for_pcg[:, :, i] = (set_cmb_x0_to_0 * Ctrue(components[:, :, i].T).T) + (np.random.randn(3, 12*nside**2)*sig_x0)
    elif comp_name[i] == 'DUST':
        #comp_for_pcg[:, :, i] = Ctrue(components[:, :, i].T).T
        comp_for_pcg[:, :, i] = HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(0))(components[:, :, i].T).T
    elif comp_name[i] == 'SYNCHROTRON':
        comp_for_pcg[:, :, i] = 0#HealpixConvolutionGaussianOperator(fwhm=np.deg2rad(2))(components[:, :, i].T).T
    elif comp_name[i] == 'CO':
        comp_for_pcg[:, :, i] = 0
    else:
        raise TypeError(f'{comp_name[i]} not recognize')
    
#comp_for_pcg[:, pixok, :] = 0
#########################################################################################################
############################################## Main Loop ################################################
#########################################################################################################


kmax=3000
k=0
g_i = np.ones((myqubic.number_FP, myqubic.Ndets))
beta_i = beta.copy()#np.ones((12*nside_fit**2, nb_param))
#if dust[0].lower() == 'true':
#    beta_i[index_seenpix, i_d-1] *= 1.54
#if synchrotron[0].lower() == 'true':
#    beta_i[index_seenpix, i_sync-1] *= -3

components_i = comp_for_pcg.copy()

def chi2_wide(x, patch_id, allbeta, solution):

    """
    
    Define chi^2 function for Wide Band TOD with shape :

        chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

    """
    allbeta[patch_id] = x
    tod_s_i = tod_w.copy() * 0
    R = ReshapeOperator(((12*nside**2,1,3)), ((12*nside**2,3)))
    #G_w = DiagonalOperator(gw, broadcast='rightward')

    k=0
    for ii, i in enumerate(array_of_operators[:2*nsub]):
    
        A = Acq.get_mixing_operator(allbeta, nus=np.array([myqubic.allnus[k]]), comp=comp, nside=nside, active=False)
        Hi = i.copy()
        Hi.operands[-1] = A
            
        tod_s_i += Hi(solution[ii]).ravel()
        k+=1
    
    if nu_co is not None:
        A = Acq.get_mixing_operator(x, nus=np.array([nu_co]), comp=comp, nside=nside, active=True)
        Hi = array_of_operators[-1].copy()
        Hi.operands[-1] = A

        tod_s_i += Hi(solution[-1]).ravel()

    
    tod_150_norm = tod_w/tod_w.max()
    tod_s_i_norm = tod_s_i/tod_s_i.max()

    return np.sum((tod_150_norm - tod_s_i_norm)**2)
def chi2_150(x, patch_id, allbeta, solution):

    """
    
    Define chi^2 function for 150 GHz TOD with shape :

        chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

    """
    allbeta[patch_id] = x
    tod_s_i = tod_150.copy() * 0
    R = ReshapeOperator(((12*nside**2,1,3)), ((12*nside**2,3)))
    #G150 = DiagonalOperator(g150, broadcast='rightward')
    k=0
    #print('len 150 = ', len(array_of_operators150))
    for ii, i in enumerate(array_of_operators150):
        #print(ii)
        A = Acq.get_mixing_operator(allbeta, nus=np.array([myqubic.allnus[k]]), comp=comp, nside=nside, active=False)
        Hi = i.copy()
        Hi.operands[-1] = A
        
        tod_s_i += Hi(solution[ii]).ravel()
        k+=1

    
    tod_150_norm = tod_150/tod_150.max()#/np.std(tod_150)
    tod_s_i_norm = tod_s_i/tod_s_i.max()#/np.std(tod_s_i)

    return np.sum((tod_150_norm - tod_s_i_norm)**2)
def chi2_220(x, patch_id, allbeta, solution):

    """
    
    Define chi^2 function for 220 GHz TOD with shape :

        chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

    """
    allbeta[patch_id] = x
    #G220 = DiagonalOperator(g220, broadcast='rightward')
    tod_s_ii = tod_220.copy() * 0
    R = ReshapeOperator(((12*nside**2,1,3)), ((12*nside**2,3)))

    k=0
    for ii, i in enumerate(array_of_operators220):
        mynus = np.array([myqubic.allnus[k+int(nsub)]])
        A = Acq.get_mixing_operator(allbeta, nus=mynus, comp=comp, nside=nside, active=False)
        Hi = i.copy()
        Hi.operands[-1] = A
        tod_s_ii += Hi(solution[ii+int(nsub)]).ravel()
        k+=1
        

    tod_220_norm = tod_220/tod_220.max()
    tod_s_ii_norm = tod_s_ii/tod_s_ii.max()
    return np.sum((tod_220_norm - tod_s_ii_norm)**2)
def chi2_external(x, patch_id, allbeta, solution):

    """
    
    Define chi^2 function for external data with shape :

        chi^2 = (TOD_true - sum_nsub(H * A * c))^2 

    """
    allbeta[patch_id] = x
    tod_s_i = tod_external.copy() * 0

    Hexternal = Acq.OtherDataParametric(external, nside, comp, nintegr=nintegr).get_operator(beta=allbeta, convolution=False, myfwhm=None, nu_co=nu_co)

    tod_s_i = Hexternal(solution[-1])

    
    tod_external_norm = tod_external#CMM.normalize_tod(tod_external, external, 12*nside**2)
    tod_s_i_norm = tod_s_i#CMM.normalize_tod(tod_s_i, external, 12*nside**2)

    diff = tod_external_norm - tod_s_i_norm
    return np.sum((diff)**2 / np.std(tod_s_i_norm)**2)
def chi2_tot(x, patch_id, allbeta, solution):

    """
    
    Define chi^2 function for all experience :

        chi^2 = chi^2_150 + chi^2_220 + chi^2_external

    """
    allbeta[patch_id] = x
    xi2_external = chi2_external(x, patch_id, allbeta, solution)
    if type == 'Two':
        xi2_150 = chi2_150(x, patch_id, allbeta, solution)
        xi2_220 = chi2_220(x, patch_id, allbeta, solution)
        #print(xi2_150, xi2_220, xi2_external)
        return xi2_150 + xi2_220 + xi2_external
    elif type == 'Wide':
        xi2_w = chi2_wide(x, patch_id, allbeta, solution)
        return xi2_w + xi2_external
    
if save_each_ite is not None:
                
    dict_i = {'maps':components, 'initial':comp_for_pcg, 'beta':beta, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':1, 'execution_time':0}

    #output = open(path_to_save+'/Iter0_maps_beta_gain_rms_maps.pkl', 'wb')
    #pickle.dump(dict_i, output)
    #output.close()

del H
gc.collect()



while k < kmax :

    if rank == 0:
        print('\n#####################################')
        print(f'##########  Iteration {k+1}  ############')
        print('#####################################\n')

    #####################################
    ######## Pixels minimization ########
    #####################################

    H_i = allexp.update_A(Hrecon, beta_i)
    #print(H_i.operands)
    #if type == 'Two':
    #    Gp150 = DiagonalOperator(1/g_i[0], broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
    #    Gp220 = DiagonalOperator(1/g_i[1], broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
    #    tod[:(myqubic.Ndets*myqubic.Nsamples)] = _r.T(Gp150(_r(tod[:(myqubic.Ndets*myqubic.Nsamples)])))
    #    tod[(myqubic.Ndets*myqubic.Nsamples):(myqubic.Ndets*myqubic.Nsamples*2)] = _r.T(Gp220(_r(tod[(myqubic.Ndets*myqubic.Nsamples):(myqubic.Ndets*myqubic.Nsamples*2)])))
    #elif type == 'Wide':
    #    Gp = DiagonalOperator(1/g_i[0], broadcast='rightward', shapein=(myqubic.Ndets, myqubic.Nsamples))
    #    tod[:(myqubic.Ndets*myqubic.Nsamples)] = _r.T(Gp(_r(tod[:(myqubic.Ndets*myqubic.Nsamples)])))
        
    A = H_i.T * invN * H_i
    b = H_i.T * invN * tod
    
    if size != 1:
        comm.Barrier()

    if rank == 0:
        print('\nFitting components...\n')
    ### PCG
    solution = pcg(A, b, M=M, tol=float(tol), x0=components_i, maxiter=int(maxite), disp=True)
    
    if size != 1:
        comm.Barrier()
    if rank == 0:
        if doplot:

            #if k == 0:
            #    os.makedirs('images')
            for i in range(len(comp)):
                plt.figure(figsize=(15, 5))
                C_reconv = HealpixConvolutionGaussianOperator(fwhm=np.sqrt(myqubic.allfwhm[0]**2 - myqubic.allfwhm[-1]**2))
                if convolution:
                    C = HealpixConvolutionGaussianOperator(fwhm=myqubic.allfwhm[-1])
                else:
                    C = HealpixConvolutionGaussianOperator(fwhm=0)
                hp.gnomview(components[1, :, i], rot=center, reso=25, cmap='jet', min=-6, max=6, sub=(1, 3, 1))
                hp.gnomview(solution['x'][1, :, i], rot=center, reso=25, cmap='jet', min=-6, max=6, sub=(1, 3, 2))
                hp.gnomview(solution['x'][1, :, i] - components[1, :, i], rot=center, reso=25, cmap='jet', min=-0.1, max=0.1, sub=(1, 3, 3))
                #plt.savefig(f'images/{i}_{type}_Iter{k+1}.png')
                plt.close()

    ### Compute spectra
    components_i = solution['x'].copy()
    components_for_beta = np.zeros((2*nsub, 3, 12*nside**2, len(comp)))
    
    ### We make the convolution before beta estimation to speed up the code, we avoid to make all the convolution at each iteration
    for i in range(2*nsub):
        for jcomp in range(len(comp)):
            if convolution:
                C = HealpixConvolutionGaussianOperator(fwhm = myfwhm[i], lmax=2*nside-1)
            else:
                C = HealpixConvolutionGaussianOperator(fwhm = 0, lmax=2*nside-1)
            components_for_beta[i, :, :, jcomp] = C(components_i[:, :, jcomp].T).T

    ###################################
    ######## Gain minimization ########
    ###################################
    
    if rank == 0:
        print('\nFitting gains...\n')
    #if type == 'Two':
    #    
    #    tod150_i = _r(H_i(components_i).ravel()[:(myqubic.Ndets*myqubic.Nsamples)])
    #    tod220_i = _r(H_i(components_i).ravel()[(myqubic.Ndets*myqubic.Nsamples):2*(myqubic.Ndets*myqubic.Nsamples)])
        
    #    g150_est = give_me_intercal(tod150_i, _r(tod_150))
    #    g220_est = give_me_intercal(tod220_i, _r(tod_220))
    #    g150_est /= g150_est[0]
    #    g220_est /= g220_est[0]

    #    g_i = np.array([g150_est, g220_est])

    #elif type == 'Wide':
        
    #    todw_i = _r(H_i(components_i).ravel()[:(myqubic.Ndets*myqubic.Nsamples)])
        
    #    gw_est = give_me_intercal(todw_i, _r(tod_w))
    #    gw_est /= gw_est[0]
        
    #    g_i = np.array([gw_est])
    #    print(g_i[0, :5])

    ###################################
    ######## Beta minimization ########
    ###################################
    if rank == 0:
        print('\nFitting spectral indices...\n')
    #chi2 = partial(chi2_tot, allbeta=beta_i, solution=components_for_beta, g150=g_i[0], g220=g_i[1])
    ### We define new chi^2 function for beta knowing the components at iteration i
    if type == 'Wide':
        chi2 = partial(chi2_tot, allbeta=beta_i, solution=components_for_beta)
    elif type == 'Two':
        chi2 = partial(chi2_tot, allbeta=beta_i, solution=components_for_beta)

    ### Doing minimization
    if size != 1:
        comm.Barrier()

    if nb_param == 1:
        if dust[0].lower() == 'true':
            x0 = np.array([beta_d_mean])
        elif synchrotron[0].lower() == 'true':
            x0 = np.array([beta_s_mean])
    elif nb_param > 1:
        x0 = np.array([beta_d_mean, beta_s_mean])

    if size != 1:
        comm.Barrier()

    wrap = solver4mpi.minimizer_multiprocess.WrapperMPI(comm, chi2, x0=np.ones(1)*1.5, method='TNC', tol=1e-10, options={}, verbose=True)
    beta_i_index = wrap(index_seenpix)
    #wrap = solver4mpi.minimizer_multiprocess.DistributeMPI(comm, 2, chi2, x0=np.ones(1)*1.5, method='TNC', tol=1e-10)
    #beta_i_index = wrap.run(index_seenpix)

    #if rank == 0:
    #    beta_i_index = solver4mpi.minimizer_multiprocess.WrapperCPU(chi2, x0=x0, nproc=omp_num_threads, verbose=True, method=method, 
    #                                                                                             tol=tol_beta).perform(index_seenpix)
    #else:
    #    beta_i_index = None
    #comm.Barrier()
    
    ### Comunicate beta results to all MPI ranks
    if size != 1:
        beta_i_index = comm.bcast(beta_i_index, root=0)
    
    if rank == 0:
        print('Estimated -> ', beta_i_index)
        print('Input -> ', beta[index_seenpix, 0])
    
    if nb_param == 1:
        beta_i[index_seenpix, 0] = beta_i_index
    else:
        beta_i[index_seenpix, :] = beta_i_index
    stop
    #if rank == 0:
    #    plt.figure()
    #    hp.mollview(beta_i[:, 0], min=1.4, max=1.6, sub=(1, 3, 1))
    #    hp.mollview(beta[:, 0], min=1.4, max=1.6, sub=(1, 3, 2))
    #    hp.mollview(beta_i[:, 0] - beta[:, 0], min=-0.1, max=0.1, sub=(1, 3, 3))
    #    plt.show()

    #stop
    ### Saving components, beta, gain, convergence, etc.. for each iteration
    if rank == 0:
        if save_each_ite is not None:

            if save_last_ite:
                if k != 0:
                    os.remove(path_to_save+'/Iter{}_maps_beta_gain_rms_maps.pkl'.format(k))

            dict_i = {'maps':components_i, 'beta':beta_i, 'allfwhm':myqubic.allfwhm, 'coverage':coverage, 'convergence':solution['error']}
    
            #output = open(path_to_save+'/Iter{}_maps_beta_gain_rms_maps.pkl'.format(k+1), 'wb')
            #pickle.dump(dict_i, output)
            #output.close()


    k+=1



