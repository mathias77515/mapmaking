#########################################################################################################################
#########################################################################################################################
#########                                                                                                       #########
#########      This script allow to perform the frequency map-making for QUBIC experiments. We show here how to #########
#########      take into account the angular resolution and the bandpass mismatch in the case of the Wide Band  #########
#########      instrument.                                                                                      #########
#########                                                                                                       #########
#########################################################################################################################
#########################################################################################################################

### General importations
import numpy as np
import healpy as hp
import pysm3.units as u
from pysm3 import utils
import sys

import os
import os.path as op

path = os.path.dirname(os.getcwd()) + '/data/'
sys.path.append(os.path.dirname(os.getcwd()))

import configparser
import pickle
import matplotlib.pyplot as plt
import time
import qubic
import component_model as c
import mixing_matrix as mm
import systematics as acq
from importlib import reload

### QUBIC packages
import qubic
import frequency_acquisition as Acq
from noise_timeline import QubicNoise, QubicWideBandNoise, QubicDualBandNoise
from planck_timeline import ExternalData2Timeline

### PyOperators
from pyoperators import *
### PySimulators
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

### Modfied PCG to display execution time
from cg import pcg
def load_config(config_file):

    '''
    Read the configuration file.
    '''

    # Créer un objet ConfigParser
    config = configparser.ConfigParser()

    # Lire le fichier de configuration
    config.read(config_file)

    k = 0
    for section in config.sections():
        for option in config.options(section):
            
            # Récupérer la valeur de chaque option de configuration
            value = config.get(section, option)

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

# Read the central nu and bandwidth
nu_ave, delta_nu_over_nu = get_ultrawideband_config()

### Variables
# CMB seed
seed = int(sys.argv[1])
# if many jobs, the number of the iteration
iteration = int(sys.argv[2])
# 1 if detector noise
ndet = int(sys.argv[3])
# 1 if photon noise at 150 GHz
npho150 = int(sys.argv[4])
# 1 if photon noise at 220 GHz
npho220 = int(sys.argv[5])

ndet, npho150, npho220 = np.array([ndet, npho150, npho220], dtype=bool)


### MPI common arguments
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print(f'You resquested for {size} processus.')


### Read configuration file
load_config('config.ini')
relative_bandwidth = 0.25

### Configuration panel
if rank == 0:
    print('Configuration of the simulation :\n')
    print(f'   Nside            : {nside}')
    print(f'   Npointing        : {npointings}')
    print(f'   Nrec             : {nrec}')
    print(f'   Nsub             : {nsub}')
    print(f'   Seed CMB         : {seed}')
    print(f'   Iteration CMB    : {iteration}')
    print('\nNoise configuration  :\n')
    print(f'   Detector noise   : {ndet}')
    print(f'   Photon noise 150 : {npho150}')
    print(f'   Photon noise 220 : {npho220}')

###############################################################
########################## Dictionary #########################
###############################################################

### We define components
skyconfig = {}
if dust:
    skyconfig['dust'] = 'd0'
if cmb:
    skyconfig['cmb'] = seed


f = int(nsub/nrec)
### Dictionary for reconstruction
d = get_dict({'npointings':npointings, 'nf_recon':nrec, 'nf_sub':nsub, 'nside':nside, 'MultiBand':True, 'period':1, 'RA_center':0, 'DEC_center':-57,
              'filter_nu':nu_ave*1e9, 'noiseless':False, 'comm':comm, 'nprocs_sampling':1, 'nprocs_instrument':size,
              'photon_noise':True, 'nhwp_angles':3, 'effective_duration':3, 'filter_relative_bandwidth':delta_nu_over_nu, 
              'type_instrument':'wide', 'TemperatureAtmosphere150':None, 'TemperatureAtmosphere220':None,
              'EmissivityAtmosphere150':None, 'EmissivityAtmosphere220':None, 'detector_nep':float(detector_nep)})

### Dictionary for noise generation
dmono = get_dict({'npointings':npointings, 'nf_recon':1, 'nf_sub':1, 'nside':nside, 'MultiBand':True, 'period':1,
              'filter_nu':nu_ave*1e9, 'noiseless':False, 'comm':comm, 'nprocs_sampling':1, 'nprocs_instrument':size,
              'photon_noise':True, 'nhwp_angles':3, 'effective_duration':3, 'filter_relative_bandwidth':delta_nu_over_nu, 
              'type_instrument':'wide', 'TemperatureAtmosphere150':None, 'TemperatureAtmosphere220':None,
              'EmissivityAtmosphere150':None, 'EmissivityAtmosphere220':None, 'detector_nep':float(detector_nep)})



### Joint acquisition
joint = acq.JointAcquisitionFrequencyMapMaking(d, type, nrec, nsub)
### Joint acquisition for TOD making
joint_tod = acq.JointAcquisitionFrequencyMapMaking(d, type, nsub, nsub)
coverage = joint.qubic.subacqs[0].get_coverage()
covnorm = coverage / coverage.max()
seenpix = covnorm > covcut
### Seen pixels with planck weight = 0
seenpix_planck = covnorm > covcut_planck
mask = np.ones(seenpix.shape)
mask[seenpix_planck] /= w_planck

### Define FWHMs
if convolution:
    
    allfwhm = joint.qubic.allfwhm
    targets = np.array([])
    for irec in range(nrec):
        targets = np.append(targets, np.sqrt(allfwhm[irec*f:(irec+1)*f]**2 - np.min(allfwhm[irec*f:(irec+1)*f])**2))
    #targets = np.sqrt(allfwhm**2 - np.min(allfwhm)**2)
else:
    targets = None
    allfwhm = None

#####################################################################################
#################################### Operators ######################################
#####################################################################################

### Define reconstructed and TOD operator
H = joint.get_operator(fwhm=targets)
Htod = joint_tod.get_operator(fwhm=allfwhm)
Hqtod = joint_tod.qubic.get_operator(fwhm=allfwhm)

##########################################################################################
#################################### Simulated maps ######################################
##########################################################################################

### Sub-maps
m_nu = ExternalData2Timeline(skyconfig, joint.qubic.allnus, nrec, nside=nside, corrected_bandpass=bandpass_correction).m_nu

### Frequency maps
mean_sky = ExternalData2Timeline(skyconfig, joint.qubic.allnus, nrec, nside=nside, corrected_bandpass=bandpass_correction).maps

npho = False
if npho150 == True or npho220 == True:
    npho = True

#######################################################################################
#################################### Planck data ######################################
#######################################################################################

planck_acquisition143 = acq.PlanckAcquisition(143, joint.qubic.scene)
planck_acquisition217 = acq.PlanckAcquisition(217, joint.qubic.scene)
seed_noise_planck = 42


npl143 = planck_acquisition143.get_noise(seed_noise_planck) * level_planck_noise
npl217 = planck_acquisition217.get_noise(seed_noise_planck) * level_planck_noise

if type == 'wide':

    nq = QubicWideBandNoise(d, npointings, d['detector_nep']).total_noise(int(ndet), int(npho150), int(npho220)).ravel()
    
    if nrec != 1:
        TOD_PLANCK = np.zeros((nrec, 12*nside**2, 3))
        for irec in range(int(nrec/2)):
            if convolution:
                C = HealpixConvolutionGaussianOperator(fwhm=np.min(allfwhm[irec*f:(irec+1)*f]))
            else:
                C = HealpixConvolutionGaussianOperator(fwhm=0)
        
            TOD_PLANCK[irec] = C(mean_sky[irec] + npl143)

        for irec in range(int(nrec/2), nrec):
            if convolution:
                C = HealpixConvolutionGaussianOperator(fwhm=np.min(allfwhm[irec*f:(irec+1)*f]))
            else:
                C = HealpixConvolutionGaussianOperator(fwhm=0)
        
            TOD_PLANCK[irec] = C(mean_sky[irec] + npl217)
        
    else:
        TOD_PLANCK = np.zeros((2*nrec, 12*nside**2, 3))
        if convolution:
            C = HealpixConvolutionGaussianOperator(fwhm=allfwhm[-1])
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)

        TOD_PLANCK[0] = C(mean_sky[0] + npl143)
        TOD_PLANCK[1] = C(mean_sky[0] + npl217)

    TOD_PLANCK = TOD_PLANCK.ravel()
    TOD_QUBIC = Hqtod(m_nu).ravel() + nq
    TOD = np.r_[TOD_QUBIC, TOD_PLANCK]
else:
    nq = QubicDualBandNoise(d, npointings, detector_nep=d['detector_nep']).total_noise(int(ndet), int(npho150), int(npho220)).ravel()
    sh_q = joint.qubic.ndets * joint.qubic.nsamples
    TOD_QUBIC = Hqtod(m_nu).ravel() + nq

    TOD_QUBIC150 = TOD_QUBIC[:sh_q].copy()
    TOD_QUBIC220 = TOD_QUBIC[sh_q:].copy()
    
    #nq150 = QubicNoise(150, npointings).total_noise(int(ndet), int(npho)).ravel()
    #nq220 = QubicNoise(220, npointings).total_noise(int(ndet), int(npho)).ravel()

    TOD = TOD_QUBIC150.copy()
    
    TOD_PLANCK = np.zeros((nrec, 12*nside**2, 3))
    for irec in range(int(nrec/2)):
        if convolution:
            C = HealpixConvolutionGaussianOperator(fwhm=np.min(allfwhm[irec*f:(irec+1)*f]))
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)
        
        #TOD_PLANCK[irec] = C(mean_sky[irec] + npl143)
        TOD = np.r_[TOD, C(mean_sky[irec] + npl143).ravel()]

    
    TOD = np.r_[TOD, TOD_QUBIC220.copy()]
    for irec in range(int(nrec/2), nrec):
        if convolution:
            C = HealpixConvolutionGaussianOperator(fwhm=np.min(allfwhm[irec*f:(irec+1)*f]))
        else:
            C = HealpixConvolutionGaussianOperator(fwhm=0)
        
        #TOD_PLANCK[irec] = C(mean_sky[irec] + npl217)
        TOD = np.r_[TOD, C(mean_sky[irec] + npl217).ravel()]


### Wait for all processus
comm.Barrier()

### Inverse noise covariance matrix
invN = joint.get_invntt_operator(mask=mask)

#R = ReshapeOperator((1, 12*nside**2, 3), (12*nside**2, 3))
### Unpack Operator to fix some pixel during the PCG
#U = (
#    ReshapeOperator((nrec * sum(seenpix) * 3), (nrec, sum(seenpix), 3)) *
#    PackOperator(np.broadcast_to(seenpix[None, :, None], (nrec, seenpix.size, 3)).copy())
#).T

### Compute A and b
#with rule_manager(none=True):
#    if nrec == 1:
#        A = U.T * R.T * H.T * invN * H * R * U
#        x_planck = mean_sky * (1 - seenpix[None, :, None])
#        b = U.T ( R.T * H.T * invN * (TOD - H(R(x_planck))))
#    else:
#        A = U.T * H.T * invN * H * U
#        x_planck = mean_sky * (1 - seenpix[None, :, None])
#        b = U.T (  H.T * invN * (TOD - H(x_planck)))
A = H.T * invN * H
b = H.T * invN * TOD

### Preconditionning
M = Acq.get_preconditioner(np.ones(12*nside**2))

### PCG
if rank == 0:
    print('\n***** PCG ******\n')


### PCG
start = time.time()
solution_qubic_planck = pcg(A, b, x0=None, M=M, tol=1e-25, disp=True, maxiter=maxiter)
end = time.time()
execution_time = end - start

#mysolution = mean_sky.copy()
#mysolution[:, ~seenpix_planck] = hp.UNSEEN
#mean_sky[:, ~seenpix] = hp.UNSEEN
#if nrec == 1:
#    mysolution[:, seenpix] = solution_qubic_planck['x'].copy()
#else:
#    mysolution[:, seenpix] = solution_qubic_planck['x'].copy()

if nrec == 1:
    mysolution = solution_qubic_planck['x'].copy()
else:
    mysolution = solution_qubic_planck['x'].copy()

center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
if doplot:
    if rank == 0:
        if convolution:
            Creconv = HealpixConvolutionGaussianOperator(fwhm=np.max(joint.qubic.allfwhm))
            C = HealpixConvolutionGaussianOperator(fwhm=np.min(joint.qubic.allfwhm))
        else:
            Creconv = HealpixConvolutionGaussianOperator(fwhm=np.max(joint.qubic.allfwhm))
            C = HealpixConvolutionGaussianOperator(fwhm=0)
        
        if nrec == 1:
            plt.figure(figsize=(8, 5))

            hp.gnomview(Creconv(mysolution[0, :, 1]), min=-8, max=8, cmap='jet', sub=(1, 3, 1), rot=center, reso=15)
            hp.gnomview(Creconv(C(mean_sky[0, :, 1])), min=-8, max=8, cmap='jet', sub=(1, 3, 2), rot=center, reso=15)
            res = Creconv(mysolution[0, :, 1])-Creconv(C(mean_sky[0, :, 1]))
            res[~seenpix] = hp.UNSEEN
            hp.gnomview(res, min=-1, max=1, cmap='jet', sub=(1, 3, 3), rot=center, reso=15)
            plt.savefig(f'band{type}_ndet{ndet}_npho150{npho150}_npho220{npho220}_{seed}_{iteration}.png')
            plt.close()
        else:
            plt.figure(figsize=(8, 5))
            k = 0
            for irec in range(nrec):
                k+=1
                hp.gnomview(Creconv(mysolution[irec, :, 1]), min=-20, max=20, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
                k+=1
                hp.gnomview(Creconv(C(mean_sky[irec, :, 1])), min=-20, max=20, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
                k+=1
                res = Creconv(mysolution[irec, :, 1])-Creconv(C(mean_sky[irec, :, 1]))
                res[~seenpix] = hp.UNSEEN
                hp.gnomview(res, min=-8, max=8, cmap='jet', sub=(nrec, 3, k), rot=center, reso=15)
            plt.savefig(f'band{type}_ndet{ndet}_npho150{npho150}_npho220{npho220}_{seed}_{iteration}.png')
            plt.close()




if rank == 0:
    print(f'Simulation done in {execution_time} s')

dict_i = {'output':mysolution, 'input':mean_sky, 'allfwhm':allfwhm, 'coverage':coverage, 'center':center, 'nsub':nsub, 'nrec':nrec, 
          'covcut':covcut, 'covcut_planck':covcut_planck, 'execution_time':execution_time, 'size':size, 'allnus':joint.qubic.allnus}



### If the folder is not here, you will create it
if rank == 0:
    save_each_ite = f'band{type}'
    current_path = os.getcwd() + '/'
    if not os.path.exists(current_path + save_each_ite):
        os.makedirs(current_path + save_each_ite)

    fullpath = current_path + save_each_ite + '/'
    output = open(fullpath+f'MM_maxiter{maxiter}_convolution{convolution}_npointing{npointings}_nrec{nrec}_nsub{nsub}_ndet{ndet}_npho150{npho150}_npho220{npho220}_seed{seed}_iteration{iteration}.pkl', 'wb')
    pickle.dump(dict_i, output)
    output.close()