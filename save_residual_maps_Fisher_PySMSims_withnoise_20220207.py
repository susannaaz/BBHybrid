import numpy as np
import healpy as hp
import glob
import os
import sys
sys.path.append('BFoRe_py/')

import bfore.maplike as mpl
import bfore.skymodel as sky
import bfore.instrumentmodel as ins
from bfore.sampling import clean_pixels, run_emcee, run_minimize, run_fisher

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--fsky', dest='fsky', default=0.1, type=float,
                  help='Set to define mask fsky')
(o, args) = parser.parse_args()

masked = True
mask_lat = False
mask_pysm = False #True
mask_planck = False
fsky = o.fsky

nsims = 5 #21
nside = 256
fdir = '/mnt/extraspace/susanna/BBSims/realistic/'
#fdir = '/mnt/extraspace/susanna/BBSims/HensleyDraine/'
#fdir = '/mnt/extraspace/susanna/BBSims/VanSyngel/'
#fdir = '/mnt/extraspace/susanna/BBSims/CurvedPlaw/'

if masked:
    if mask_lat:
        maskdir = '/mnt/zfsusers/susanna/BBSims/data/mask_apo_fsky%.1f.fits'%fsky
    elif mask_pysm:
        maskdir = '/mnt/zfsusers/susanna/BBSims/data/mask_pysm/mask_apo_pysmd0s0_fsky%.1f.fits'%fsky
    elif mask_planck:
        maskdir = '/mnt/zfsusers/susanna/BBSims/data/mask_plancktmp/mask_apo_plancktmp_fsky%.1f.fits'%fsky
    else:
        maskdir = 'mask_apodized.fits'
    print()
    print('Mask used:')
    print(maskdir)
    print()
    sname = 'masked'
    sat_mask = hp.ud_grade(hp.read_map(maskdir), nside)
else:
    sname = 'full'
    sat_mask = np.ones(hp.nside2npix(nside))
npix = np.sum(sat_mask>0)

# Generate N matrix
def get_noise(dirs):
    split1 = hp.read_map(f'{dirs}obs_split1of4.fits.gz', field=np.arange(12), verbose=False)
    split2 = hp.read_map(f'{dirs}obs_split2of4.fits.gz', field=np.arange(12), verbose=False)
    dnoise = split1 - split2
    Qn = dnoise[::2, sat_mask>0]
    Un = dnoise[1::2, sat_mask>0]
    noimaps = np.array([np.transpose(Qn), np.transpose(Un)]) # npol, npix, nfreq
    return Qn, Un, noimaps

def noivar(data):
    """
    Generate variance and inverse-variance of each frequency.
    where  Variance = standard deviation over the last axis 
    (i.e. over a whole pixel) squared.
    
    This gives the pixel-independent part of the covariance
    matrix that you want to pass to BFore,
    i.e. assumes it is homogeneous across the sky
    i.e. can be used to generate Q matrix 
         where the inhomogeneous part is factorised out.
    
    Args:
      data: [N_freq, N_pix]

    Returns:
      nvar: Variance of the data. 
      nvar_inv: Inverse variance of the data. 
    """
    nfreq, npix = data.shape
    nVar = np.std(data, axis=-1)**2
    print(nVar.shape)
    nVar_inv = 1/nVar
    return nVar, nVar_inv

def coadd_maps(splits):
    """ Generate the coadded splits map.
    
    Args:
      splits: list of observation splits
    
    Returns:
      coadd: sum_over_splits of the maps_of_each_splits divided by number of splits.
             i.e. 12 maps for each split (Q/U * nfreq)
    """
    s0 = hp.read_map(splits[0], field=np.arange(12), verbose=False)
    s1 = hp.read_map(splits[1], field=np.arange(12), verbose=False)
    s2 = hp.read_map(splits[2], field=np.arange(12), verbose=False)
    s3 = hp.read_map(splits[3], field=np.arange(12), verbose=False)
    coadd = (s0 + s1 + s2 + s3)/4
    return coadd
    
def get_params(k, sdir):
    """ Generate the residual amplitudes in each observation split. 

    Args:
      k: simulation number [int]
      split: observation split [str]
      split_number: [int]
      fn: directory containing the simulation [str]
      sn: output directory [str]

    Returns:
      Saves the maximum likelihood betas (params_cent), their sigmas (params_sigma),
      mixing matrix (Sbar) and Q matrix.
    """
    testmap = coadd_maps(splits) #[N_freq, N_pix]
    Qs = testmap[::2, sat_mask>0]
    Us = testmap[1::2, sat_mask>0]
    skymaps = np.array([np.transpose(Qs), np.transpose(Us)]) #[N_pol,N_pix,N_freq]

    Qn, Un, noise = get_noise(fn) 
    Qn_sigmas, Qninv_sigmas = noivar(Qn) #get inverse variance of noise
    Ninv = np.diag(Qninv_sigmas) #get diagonal matrix
    
    nside = 256
    if mask_pysm:
        nhits = hp.ud_grade(hp.read_map(maskdir, verbose=False), nside_out=nside)
    else:
        nhits=hp.ud_grade(hp.read_map("./data/norm_nHits_SA_35FOV.fits", verbose=False), nside_out=nside)

    nhits[nhits < 1E-3] = 1E-3
    nhits = nhits[sat_mask > 0]
    ii = np.ones((2, len(nhits)))
    
    p = (np.array([ii / (nhits * sig) for sig in Qninv_sigmas]))

    Ninv_sp = np.transpose(p, axes=[1,2,0]) #npol, npix, nfreq

    #array([-3.08336434,  1.62253067]) --> sync, dust
    
    nu_ref_sync_p=23.
    beta_sync_fid=-3.
    curv_sync_fid=0.

    nu_ref_dust_p=353.
    beta_dust_fid=1.5
    temp_dust_fid=19.6

    spec_i=np.zeros([2, npix]);

    bs=beta_sync_fid; bd=beta_dust_fid; td=temp_dust_fid; cs=curv_sync_fid;
    sbs=3.0; sbd=3.0; 
    spec_i[0]=bs; spec_i[1]=bd

    fixed_pars={'nu_ref_d':nu_ref_dust_p,'nu_ref_s':nu_ref_sync_p,'T_d':td}
    var_pars=['beta_s','beta_d']
    var_prior_mean=[bs,bd]
    var_prior_width=[sbs,sbd]

    sky_true=sky.SkyModel(['syncpl', 'dustmbb', 'unit_response'])
    nus = [27., 39., 93., 145., 225., 280.]
    bps=np.array([{'nu':np.array([n-0.5,n+0.5]),'bps':np.array([1])} for n in nus])
    instrument=ins.InstrumentModel(bps)
    ml=mpl.MapLike({'data': skymaps, #coadded
                    'noisevar': Ninv_sp, 
                    'fixed_pars':fixed_pars,
                    'var_pars':var_pars,
                    'var_prior_mean':var_prior_mean,
                    'var_prior_width':var_prior_width,
                    'var_prior_type':['tophat' for b in var_pars]}, 
                   sky_true, 
                   instrument)
    sampler_args = {
        "ml_first" : True,
        "ml_method" : 'Powell',
        "ml_options" : {'xtol':1E-4,'ftol':1E-4,'maxiter':None,'maxfev':None,'direc':None}
        }
    # Compute best fit spectral indices and sigmas
    rdict = clean_pixels(ml, run_fisher, **sampler_args)
    params_cent = rdict['params_cent'] #Params centre (beta_d, beta_s)
    covar = np.linalg.inv(rdict['fisher_m'])
    params_sigma = np.sqrt(np.diag(covar)) #Params sigma (priors)

    print(params_cent)
    print(params_sigma)
    
    Sbar = ml.f_matrix(params_cent).T #mixing matrix for ML parameters [n_comp * n_freqs]
    
    Sninv = np.linalg.inv((Sbar.T).dot(Ninv).dot(Sbar)) 
    P = np.diag([1., 1., 0.])
    Q = np.identity(6) - Sbar.dot(P).dot(Sninv).dot(Sbar.T).dot(Ninv)

    np.savez(f'{sdir}{sname}_hybrid_params_{k}', params_cent=params_cent, params_sigma=params_sigma, Sbar=Sbar, Q=Q)
    return Q

def clean_maps(k, split, split_number, fn, sdir, Q):
    """ Generate the residual amplitudes in each observation split. 

    Args:
      k: simulation number [int]
      split: observation split [str]
      split_number: [int]
      fn: directory containing the simulation [str]
      sn: output directory [str]

    Returns:
      Saves the residual amplitude maps (filled maps)
    """
    sp = hp.read_map(split, field=np.arange(12), verbose=False)
    Qsp = sp[::2, sat_mask>0]
    Usp = sp[1::2, sat_mask>0]
    spmaps = np.array([np.transpose(Qsp), np.transpose(Usp)])

    reducedmaps = np.einsum('ab, cdb', Q, spmaps).reshape((12, -1))
    
    filled_maps = np.zeros((12, hp.nside2npix(nside))) 
    filled_maps[:, sat_mask>0] = reducedmaps
    hp.write_map(f'{sdir}{sname}_residualmaps_{k}_split%i.fits'%split_number, filled_maps, overwrite=True)
    return

# TO CHANGE
#ddir = f'sim_ns256_fullsky_E_B_cmb_dust_sync_PySMBetas_PySMAmps_nu0d353_nu0s23/'
#ddir = f'sim_ns256_fullsky_E_B_cmb_dust_sync_PySMBetas_PySMAmps_d1s1_nu0d353_nu0s23/'
#ddir = f'sim_ns256_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_d1s1_nu0d353_nu0s23/' #B only
#ddir = f'sim_ns256_r0.01_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_nu0d353_nu0s23/' #B only, r=0.01
#ddir = f'sim_ns256_r0.01_whitenoiONLY_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_nu0d353_nu0s23/' #B only, r=0.01, white_noi only
#ddir = f'sim_ns256_whitenoiONLY_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_nu0d353_nu0s23/' #B only, white_noi only
#ddir = f'sim_ns256_msk_E_B_cmb_dust_sync_GNILCbetaD_PySMbetaS_GNILCampD_PySMampS_nu0d353_nu0s23/' # GNILC sims
ddir = f'sim_ns256_fullsky_E_B_cmb_dust_sync_GNILCbetaD_PySMbetaS_PySMAmps_nu0d353_nu0s23/' # GNILC sims (beta_D only)

if masked:
    if mask_lat:
        sdir = f'./data/simd1s1_masklat/fsky%.1f/'%fsky #f'./data/simd1s1/'
    elif mask_pysm:
        # TO CHANGE
        #sdir = f'./data/simd1s1_maskpysm_Bonly/fsky%.1f/'%fsky
        #sdir = f'./data/simd1s1_maskpysm_Bonly_r0.01/fsky%.1f/'%fsky
        #sdir = f'./data/simd1s1_maskpysm_Bonly_r0.01_whitenoiONLY/fsky%.1f/'%fsky
        sdir = f'./data/simd1s1_maskpysm_Bonly_whitenoiONLY/fsky%.1f/'%fsky
    elif mask_planck:
        sdir = f'./data/simd1s1_maskplanck_Bonly/fsky%.1f/'%fsky
    else:
        #sdir = f'./data/simd1s1/'
        #sdir = f'./data/simHensleyDraine/'
        #sdir = f'./data/simVanSyngel/'
        #sdir = f'./data/simCurvedPlaw/'
        #sdir = f'./data/simGNILC/'
        sdir = f'./data/sim_GNILCbetaD_PySMAmps/'
else:
    #fsky=1.
    #sdir = f'./data/simd1s1/fsky%.1f/'%fsky
    sdir = f'./data/simd1s1_maskpysm_Bonly/fsky%.1f_fromfullmaps/'%fsky

print()
print('Output directory:')
print(sdir)
print()
os.system('mkdir -p ' + sdir)

fnames = glob.glob(f'{fdir}{ddir}s*/')
fnames.sort()
fnames = fnames[1:]
for k, fn in enumerate(fnames[:nsims]):
    splits = glob.glob(f'{fn}obs_split*of4.fits.gz')
    splits.sort()
    Q = get_params(str(k).zfill(4), sdir)
    for s, sn in enumerate(splits):
        clean_maps(str(k).zfill(4), sn, s, fn, sdir, Q)
