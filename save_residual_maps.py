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

masked = True

#nsims = 21
nsims = 1
nside = 256
fdir = '/mnt/extraspace/susanna/BBHybrid/'

if masked:
    sname = 'masked'
    sat_mask = hp.ud_grade(hp.read_map('mask_apodized.fits'), nside)
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
    
def clean_maps(k, split, split_number, fn, sdir, cleanmap):
    """ Generate the residual amplitudes in each observation split. 

    Args:
      k: simulation number [int]
      split: observation split [str]
      split_number: [int]
      fn: directory containing the simulation [str]
      sn: output directory [str]

    Returns:
      Saves the mixing matrix (Sbar) and Q matrix.
      Saves the residual amplitude maps (filled maps)
    """
    #testmap = coadd_maps(splits) #[N_freq, N_pix]
    print(cleanmap)
    testmap = hp.read_map(cleanmap, field=np.arange(12), verbose=False)
    Qs = testmap[::2, sat_mask>0]
    Us = testmap[1::2, sat_mask>0]
    skymaps = np.array([np.transpose(Qs), np.transpose(Us)]) #[N_pol,N_pix,N_freq]
    print(skymaps.shape)
    print(skymaps)

    # Noise covariance matrix, [N_freq, N_freq]
    print(fn)
    Qn, Un, noise = get_noise(fn) 
    Qn_sigmas, Qninv_sigmas = noivar(Qn) #get inverse variance of noise
    print("HIII")
    print(Qn_sigmas)
    print(Qninv_sigmas)
    Qninv_sigmas[:] = 1.
    Ninv = np.diag(Qninv_sigmas) #get diagonal matrix

    nside = 256
    nhits=hp.ud_grade(hp.read_map("./data/norm_nHits_SA_35FOV.fits", verbose=False), nside_out=nside)
    ii = np.ones((2, hp.nside2npix(nside)))
    
    p = (np.array([ii * nhits * sig for sig in Qninv_sigmas]))
    p = p[:, :, sat_mask>0] #nfreq, npol, npix
    Ninv_sp = np.transpose(p, axes=[1,2,0])
    print(Ninv_sp.shape)
    print('should be npol, npix, nfreq')

    nu_ref_sync_p=23.
    beta_sync_fid=-3.
    curv_sync_fid=0.

    nu_ref_dust_p=353.
    beta_dust_fid=1.5
    temp_dust_fid=19.6

    spec_i=np.zeros([2, npix]);
    #spec_o=np.zeros([2, npix]);
    #amps_o=np.zeros([3, 2, npix]);
    #cova_o=np.zeros([6, 2, npix]);

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
                    'noisevar': np.ones_like(skymaps), #Ninv_sp, 
                    'fixed_pars':fixed_pars,
                    'var_pars':var_pars,
                    'var_prior_mean':var_prior_mean,
                    'var_prior_width':var_prior_width,
                    'var_prior_type':['tophat' for b in var_pars]}, 
                   sky_true, 
                   instrument)
    sampler_args = {
        "method" : 'Powell',
        "tol" : None,
        "callback" : None,
        "options" : {'xtol':1E-4,'ftol':1E-4,'maxiter':None,'maxfev':None,'direc':None}
        }

    # Compute best fit spectral indices
    rdict = clean_pixels(ml, run_minimize, **sampler_args)
    print(rdict)
    print(rdict['params_ML'])
    # Compute mixing matrix [n_components * n_freqs]
    Sbar = ml.f_matrix(rdict['params_ML']).T #mixing matrix for ML parameters

    # T matrix: [n_components * n_components]

    # N matrix per pixel
    
    # (S^T * N^-1 * S)
    Sninv = np.linalg.inv((Sbar.T).dot(Ninv).dot(Sbar)) 
    # 1 - P to select only CMB
    P = np.diag([1., 1., 0.])
    # Filter
    # in Q matrix use homogeneous Nvar
    # non homogeneous Nvar factorises out
    Q = np.identity(6) - Sbar.dot(P).dot(Sninv).dot(Sbar.T).dot(Ninv)

    # Residuals for each split
    sp = hp.read_map(split, field=np.arange(12), verbose=False)
    Qsp = sp[::2, sat_mask>0]
    Usp = sp[1::2, sat_mask>0]
    spmaps = np.array([np.transpose(Qsp), np.transpose(Usp)])

    reducedmaps = np.einsum('ab, cdb', Q, spmaps).reshape((12, -1))
    
    filled_maps = np.zeros((12, hp.nside2npix(nside))) 
    filled_maps[:, sat_mask>0] = reducedmaps

    # Save the results for each split
    np.savez(f'{sdir}{sname}_hybrid_params_{k}_split%i'%split_number, params=rdict['params_ML'], Sbar=Sbar, Q=Q)
    hp.write_map(f'{sdir}{sname}_residualmaps_{k}_split%i.fits'%split_number, filled_maps, overwrite=True)
    return

# repeating BFore 4 times 1 for each split but params are the same for all splits
# only do one results save residuals and then apply to all the splits
stds = 1
for std in range(stds):
    ddir = f'sim_ns256_stdd0.{std}_stds0.{std}_gdm3.0_gsm3.0_fullsky_E_B_nu0d353_nu0s23_cmb_dust_sync_Ad28.0_As1.6_ad0.16_as0.93/'
    sdir = f'./data/sim0{std}/'
    os.system('mkdir -p ' + sdir)
    fnames = glob.glob(f'{fdir}{ddir}s*/')
    fnames.sort()
    for k, fn in enumerate(fnames[:nsims]):
        splits = glob.glob(f'{fn}obs_split*of4.fits.gz')
        splits.sort()
        clnmp = f'{fn}maps_sky_signal.fits'
        for s, sn in enumerate(splits):
            clean_maps(str(k).zfill(stds), sn, s, fn, sdir, clnmp)
            exit()
            print(s, sn)
            clean_maps(str(k).zfill(stds), sn, s, fn, sdir)


