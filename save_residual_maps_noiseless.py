import numpy as np
import healpy as hp
import glob
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
#'/mnt/extraspace/susanna/BBMoments_SO/BBMoments/'
#'/mnt/extraspace/damonge/BBHybrid/' #'/mnt/zfsusers/mabitbol/simdata/'

if masked:
    sname = 'masked'
    sat_mask = hp.ud_grade(hp.read_map('mask_apodized.fits'), nside)
else:
    sname = 'full'
    sat_mask = np.ones(hp.nside2npix(nside))
npix = np.sum(sat_mask>0)

# 0) Generate N matrix
#     split 1 – split 2 = n1 – n2 (just contain the noise)
#     take variance of noise and put in diagonal
# 1) Coadd maps
#     Instead of maps_sky_signal, use each split
#     sum_over_splits (s) of the maps_of_each_splits divided by 4
# 2) Clean_pixel from coadded maps
# 3) Work out Q matrix including N
# 4) Apply Q to each split
# 5) Save the results in 4 separate splits
def clean_maps(k, fn, sdir):
    testmap = hp.read_map(f'{fn}maps_sky_signal.fits', field=np.arange(12), verbose=False)
    Qs = testmap[::2, sat_mask>0]
    Us = testmap[1::2, sat_mask>0]
    skymaps = np.array([np.transpose(Qs), np.transpose(Us)])
    print(skymaps.shape)

    nu_ref_sync_p=23.
    beta_sync_fid=-3.
    curv_sync_fid=0.

    nu_ref_dust_p=353.
    beta_dust_fid=1.5
    temp_dust_fid=19.6

    spec_i=np.zeros([2, npix]);
    spec_o=np.zeros([2, npix]);
    amps_o=np.zeros([3, 2, npix]);
    cova_o=np.zeros([6, 2, npix]);

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
    ##### change skymaps
    ##### change noise
    ml=mpl.MapLike({'data': skymaps, 
                    'noisevar':np.ones_like(skymaps),
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

    # Compute best fit spectral indices (+ amplitudes?)
    rdict = clean_pixels(ml, run_minimize, **sampler_args)

    # Compute mixing matrix [n_components * n_freqs] 
    Sbar = ml.f_matrix(rdict['params_ML']).T

    # Find amplitudes
    #   i.e. amp = (mixing matrix S * noise covariance^-1 * S) * (S * N^-1 * data)
    ######### Need amps in each splits,
    ######### 12 maps for each split (coadded maps)
    ######### sum_over_splits (s) of the maps_of_each_splits divided by 4
    #########(it’s just reading each of them adding them together and dividing by number of splits -average- )
    #########
    #
    # (S^T * N^-1 * S), where here N^-1 is identity
    ######### NOISE COVARIANCE NOT IDENTITY
    Sninv = np.linalg.inv(np.dot(Sbar.T, Sbar))
    # 1 - P to select only CMB
    P = np.diag([1., 1., 0.])
    # Filter 
    #    Q = 1 - Sbar P Sninv S N^-1, where N^-1 is identity
    ######### NOISE COVARIANCE NOT IDENTITY
    Q = np.identity(6) - Sbar.dot(P).dot(Sninv).dot(Sbar.T)

    # Residuals
    ######### Need residuals from each split
    ######### Save residuals for each split
    ######### NOISE COVARIANCE NOT IDENTITY
    reducedmaps = np.einsum('ab, cdb', Q, skymaps).reshape((12, -1))
    
    filled_maps = np.zeros((12, hp.nside2npix(nside))) 
    filled_maps[:, sat_mask>0] = reducedmaps
    
    np.savez(f'{sdir}{sname}_hybrid_params_{k}', params=rdict['params_ML'], Sbar=Sbar, Q=Q)
    hp.write_map(f'{sdir}{sname}_residualmaps_{k}.fits', filled_maps, overwrite=True)
    return


#for std in range(4):
for std in range(1):
    #ddir = f'sims_Ad5_As2/sim_ns256_stdd{std}_stds0_gdm3.0_gsm3.0_fullsky_E_B_nu0d353_nu0s23_cmb_dust_sync_Ad5/'
    ddir = f'sim_ns256_stdd0.{std}_stds0.{std}_gdm3.0_gsm3.0_fullsky_E_B_nu0d353_nu0s23_cmb_dust_sync_Ad28.0_As1.6_ad0.16_as0.93/'
    #f'sims_gauss_fullsky_ns256_csd_std0.{std}_gm3/'
    sdir = f'./data/sim0{std}_noiseless/'

    fnames = glob.glob(f'{fdir}{ddir}s*/')
    fnames.sort()
    #print(fnames)
    for k, fn in enumerate(fnames[:nsims]):
        clean_maps(str(k).zfill(4), fn, sdir)

