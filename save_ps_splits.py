import numpy as np
from utils import *
import noise_calc as nc
import sys
sys.path.append('sacc/sacc')

import sacc
import healpy as hp
import pymaster as nmt
import glob
from optparse import OptionParser
parser = OptionParser()
parser.add_option('--fsky', dest='fsky', default=0.8, type=float,
                  help='Set to define mask fsky')
(o, args) = parser.parse_args()

"""
Note we have only B in input simulations
"""

residuals = False
masked = True
nsims = 21

whitenoi_only = True
r001 = False # if r=0.01 in input sims

fsky = o.fsky #0.8

nside = 256
npix = hp.nside2npix(nside)

nfreqs = 6
npol = 2
sens = 2
knee = 1
ylf = 1

# Bandpasses
bpss = {n: Bpass(n,f'data/bandpasses/{n}.txt') for n in band_names}

# Bandpowers
dell = 10
nbands = 76
lmax = 2+nbands*dell
larr_all = np.arange(lmax+1)
lbands = np.linspace(2, lmax, nbands+1, dtype=int)
leff = 0.5*(lbands[1:]+lbands[:-1])
windows = np.zeros([nbands, lmax+1])
for b,(l0,lf) in enumerate(zip(lbands[:-1], lbands[1:])):
    windows[b,l0:lf] = (larr_all * (larr_all + 1)/(2*np.pi))[l0:lf]
    windows[b,:] /= dell
s_wins = sacc.BandpowerWindow(larr_all, windows.T)

# Beams
beams = {band_names[i]: b for i, b in enumerate(nc.Simons_Observatory_V3_SA_beams(larr_all))}

# N_ell
nell = np.zeros([nfreqs, lmax+1])
_, nell[:, 2:], _ = nc.Simons_Observatory_V3_SA_noise(sens, knee, ylf, fsky, lmax+1, 1, whitenoi_ONLY=whitenoi_only)
n_bpw = np.sum(nell[:, None, :]*windows[None, :, :], axis=2)
bpw_freq_noi = np.zeros((nfreqs, npol, nfreqs, npol, nbands))
for ib,n in enumerate(n_bpw):
    bpw_freq_noi[ib, 0, ib, 0, :] = n_bpw[ib, :]
    if npol==2:
        bpw_freq_noi[ib, 1, ib, 1, :] = n_bpw[ib, :]
bpw_freq_noi_re = bpw_freq_noi.reshape([nfreqs*npol, nfreqs*npol, nbands])

def saveps(): #sk):
    #'data/simd1s1_maskplanck/fsky%.1f/'%fsky #'data/simd1s1/'
    #prefix_in = f'data/simd1s1_maskplanck_Bonly/fsky%.1f/'%fsky
    #prefix_in = f'data/simd1s1_maskpysm_Bonly/fsky%.1f/'%fsky
    #prefix_in = f'data/simd1s1_maskpysm_Bonly_r0.01/fsky%.1f/'%fsky
    #prefix_in = f'data/simd1s1_maskpysm_Bonly_r0.01_whitenoiONLY/fsky%.1f/'%fsky
    prefix_in = f'data/simd1s1_maskpysm_Bonly_whitenoiONLY/fsky%.1f/'%fsky
    #prefix_in = f'data/simd1s1_maskpysm_Bonly/fsky%.1f_SATmask_B/'%fsky
    #'data/knoxcov_simd1s1_maskplanck/fsky%.1f/'%fsky
    prefix_out = prefix_in
    print(prefix_in)
    print(prefix_out)

    # Loop over sims
    for kn in range(nsims):
        skn = str(kn).zfill(4) 

        if residuals:
            sname = 'residual'
            fnames = glob.glob(f'{prefix_in}masked_residualmaps*_split*.fits')
        else:
            sname = 'baseline'
            fdir = '/mnt/extraspace/susanna/BBSims/realistic/'
            if whitenoi_only:
                if r001:
                    ddir = f'sim_ns256_r0.01_whitenoiONLY_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_nu0d353_nu0s23/'
                else: #r=0
                    ddir = f'sim_ns256_whitenoiONLY_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_nu0d353_nu0s23/'
            else:
                if r001:
                    ddir = f'sim_ns256_r0.01_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_d1s1_nu0d353_nu0s23/'
                else:
                    ddir = f'sim_ns256_fullsky_B_cmb_dust_sync_PySMBetas_PySMAmps_d1s1_nu0d353_nu0s23/'
            
            fnames = glob.glob(f'{fdir}{ddir}s*/obs_split*of4.fits.gz')
        fnames.sort()

        if masked:
            sname += '_masked'
            #sat_mask = hp.ud_grade(hp.read_map('mask_apodized.fits'), nside)
            #sat_mask = hp.ud_grade(hp.read_map('/mnt/zfsusers/susanna/BBSims/data/mask_plancktmp/mask_apo_plancktmp_fsky%.1f.fits'%fsky), nside)
            sat_mask = hp.ud_grade(hp.read_map('/mnt/zfsusers/susanna/BBSims/data/mask_pysm/mask_apo_pysmd0s0_fsky%.1f.fits'%fsky), nside)
        else: 
            sat_mask = np.ones(npix)

        nsplits = 4
        
        # Precompute coupling matrix for namaster
        b = nmt.NmtBin.from_nside_linear(nside, dell, is_Dell=True)
        purify_b = False #if only B in input sim
        empty_field = nmt.NmtField(sat_mask, maps=None, spin=2, purify_b=purify_b)
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(empty_field, empty_field, b)

        # Read maps
        kk = nsplits*kn
        fknames = fnames[kk:kk+4]
       
        maps_all = np.array([hp.read_map(i, field=np.arange(nfreqs*npol), verbose=False) for i in fknames])
        maps_all = maps_all.reshape((nsplits, nfreqs, npol, -1))        
        maps_all[:, :, :, sat_mask==0] = 0.

        # Fields for each split given each map[split,freq]
        flds = []
        for isplit in range(nsplits):
            flds.append([])
            for ifreq in range(nfreqs):
                # fields for each split given each map[split,freq]
                flds[isplit].append(nmt.NmtField(sat_mask, maps_all[isplit, ifreq], purify_b=purify_b))

        # Define the binning
        ls = b.get_effective_ells()
        nbands = len(ls)

        ## All cls [4: II EE BB EB]
        #(nsplits, nbands, npol, n_bpw) #(4 76 2)
        # should be print(self.nsplits, self.nbands, self.n_bpws) 4 6 76
        #cls = np.zeros([nsplits, nfreqs, npol, nsplits, nfreqs, npol, nbands])
        #
        ## only BB
        cls = np.zeros([nsplits, nfreqs, nsplits, nfreqs, nbands])

        ## all cross cls
        for isplit1 in range(nsplits):
            for isplit2 in range(nsplits):
                for iband1 in range(nfreqs):
                    for iband2 in range(nfreqs):
                        #for ipol1 in range(npol):
                        #    for ipol2 in range(npol):
                        fld1 = flds[isplit1][iband1]
                        fld2 = flds[isplit2][iband2]
                        # Invert MCM #TT EE EB BB
                        cl = wsp.decouple_cell(nmt.compute_coupled_cell(fld1, fld2)) #[4, 76] 
                        #cls[isplit1, iband1, ipol1, isplit2, iband2, ipol2, :] = cl #[:1] #ee+bb
                        cls[isplit1, iband1, isplit2, iband2, :] = cl[3] # only bb cls
                        
                        # Put it in shape [nsplits,nsplits,nbands,2,nbands,2,nl]
                        #spectra = np.transpose(cls, axes=[0, 3, 1, 2, 4, 5, 6]) #all cls
                        spectra = np.transpose(cls, axes=[0, 2, 1, 3, 4]) #only BB

                        # Coadding
                        # Total coadding (including diagonal)
                        weights_total = np.ones(nsplits, dtype=float)/nsplits
                        spectra_total = np.einsum('i,ijkmo,j', weights_total, spectra, weights_total) #BB only
                        # Off-diagonal coadding
                        triu_mean = np.mean(spectra[np.triu_indices(nsplits, 1)], axis=0) #upper
                        tril_mean = np.mean(spectra[np.tril_indices(nsplits, -1)], axis=0) #lower
                        spectra_xcorr = 0.5*(tril_mean+triu_mean)
        
                        # Noise power spectra
                        spectra_noise = spectra_total - spectra_xcorr

                        # Add to signal
                        bpw_freq_tot = spectra_xcorr + spectra_noise
                        #bpw_freq_tot = bpw_freq_tot.reshape([nfreqs*npol, nfreqs*npol, nbands])
                        #bpw_freq_sig = spectra_xcorr.reshape([nfreqs*npol, nfreqs*npol, nbands])
                        bpw_freq_tot = bpw_freq_tot.reshape([nfreqs, nfreqs, nbands])
                        bpw_freq_sig = spectra_xcorr.reshape([nfreqs, nfreqs, nbands])

        # Creating Sacc files
        s_d = sacc.Sacc()
        s_f = sacc.Sacc()
        s_n = sacc.Sacc()

        # Adding tracers
        for ib, n in enumerate(band_names):
            bandpass = bpss[n]
            beam = beams[n]
            for s in [s_d, s_f, s_n]:
                s.add_tracer('NuMap', 'band%d' % (ib+1),
                             quantity='cmb_polarization',
                             spin=2,
                             nu=bandpass.nu,
                             bandpass=bandpass.bnu,
                             ell=larr_all,
                             beam=beam,
                             nu_unit='GHz',
                             map_unit='uK_CMB')

        # Adding power spectra
        nnpol=1 #2
        nmaps = nnpol*nfreqs
        ncross = (nmaps*(nmaps+1))//2
        indices_tr = np.triu_indices(nmaps)
        map_names = []
        for ib, n in enumerate(band_names):
            #map_names.append('band%d' % (ib+1) + '_E')
            map_names.append('band%d' % (ib+1) + '_B')
        print(nmaps)
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            n1 = map_names[i1][:-2]
            n2 = map_names[i2][:-2]
            p1 = map_names[i1][-1].lower()
            p2 = map_names[i2][-1].lower()
            cl_type = f'cl_{p1}{p2}'
            s_d.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
            s_f.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_sig[i1, i2, :], window=s_wins)
            s_n.add_ell_cl(cl_type, n1, n2, leff, bpw_freq_noi_re[i1, i2, :], window=s_wins)

        # Add covariance (Knox formula)
        cov_bpw = np.zeros([ncross, nbands, ncross, nbands])
        factor_modecount = 1./((2*leff+1)*dell*fsky)
        for ii, (i1, i2) in enumerate(zip(indices_tr[0], indices_tr[1])):
            for jj, (j1, j2) in enumerate(zip(indices_tr[0], indices_tr[1])):
                covar = (bpw_freq_tot[i1, j1, :]*bpw_freq_tot[i2, j2, :]+
                         bpw_freq_tot[i1, j2, :]*bpw_freq_tot[i2, j1, :]) * factor_modecount
                cov_bpw[ii, :, jj, :] = np.diag(covar)
        cov_bpw = cov_bpw.reshape([ncross * nbands, ncross * nbands])
        s_d.add_covariance(cov_bpw)

        # Write output
        skn = str(kn).zfill(4) 
        s_d.save_fits(f'{prefix_out}cls_coadd_{sname}_{skn}.fits', overwrite=True)
        s_f.save_fits(f'{prefix_out}cls_fid_{sname}_{skn}.fits', overwrite=True)
        s_n.save_fits(f'{prefix_out}cls_noise_{sname}_{skn}.fits', overwrite=True)
    return 

saveps()
