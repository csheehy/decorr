########
########
# Code to produce results from this paper:
# https://journals.aps.org/prd/abstract/10.1103/PhysRevD.97.043522
# https://arxiv.org/abs/1709.09729

# Decorrelation of multifrequency CMB + dust maps as a function of ell
import numpy as np
import os
import healpy as hp
import planck
import cPickle as cP
import sys
sys.path.append('/astro/u/csheehy/python/PolSpice_v03-03-02/src')
sys.path.append('/astro/u/csheehy/python/Xpol')
from ispice import ispice
import matplotlib.pyplot as plt
from glob import glob
import string
import random
from copy import deepcopy as dc
from astropy.io import fits
import scipy.special as sp
from scipy.stats import skew
from scipy.optimize import curve_fit
import scipy as sc
import xpol_wrap as xpol


arcmin2rad = np.pi/180/60
udefval=-1.6375e+30

def nanhist(x, **kwargs):
    plt.hist(x[np.isfinite(x)],**kwargs)

def concatenate_spec(f):
    """Load in and concatenate spectra files
    e.g. concatenate_spec('spec/dust1_tophat*.pickle')"""

    d = np.sort(glob(f))

    for k,val in enumerate(d):
        print(val)
        f = open(val, 'rb')
        s0 = cP.load(f)
        f.close()

        if k==0:
            s = s0
            if hasattr(s,'s'):
                s.s = [s.s]
            if hasattr(s,'n'):
                s.n = [s.n]
            if hasattr(s,'sn'):
                s.sn = [s.sn]
        else:
            if hasattr(s0,'s'):
                s.s.append(s0.s)
            if hasattr(s0,'n'):
                s.n.append(s0.n)
            if hasattr(s0,'sn'):
                s.sn.append(s0.sn)

    if hasattr(s0,'s'):
        s.s = np.array(s.s)
    if hasattr(s0,'n'):
        s.n = np.array(s.n)
    if hasattr(s0,'sn'):
        s.sn = np.array(s.sn)

    fout = d[0][0:-11] + 'xxxx.pickle'

    f = open(fout, 'wb')
    cP.dump(s, f)
    f.close()


def randstring(size=4):
    """Generate random string of size size"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def getLR():
    return ['LR16', 'LR24', 'LR33', 'LR42', 'LR53', 'LR63N', 'LR63S', 'LR63', 'LR72']

def getfinite(x):
    return(x[np.isfinite(x)])

###################
###################

class Maps(object):
    """A class to compute correlation coefficient between two maps"""
    
    def __init__(self, nside=512, comp=None, simprefix=None, reg='LR63'):
        """Map object"""
        # To the analysis at this map resolution. Downgrade maps as appropriate
        if comp is None:
            comp = ['synch','therm']

        if simprefix is None:
            simprefix = 'dust0_tophat_'

        self.nside = nside 
        self.comp = comp
        self.simprefix = simprefix
        self.reg = reg
        self.setfsky()


    def getallmaps(self):
        """Get all maps and masks"""

        # Set the real, noiseless sim, and noise maps
        self.prepmaps('all')

        # Get all masks, PIPL LR regions + a nan mask for 217 mc noise realizations
        self.prepmasks('all')


    def setfsky(self):
        """Get the fsky, same as self.reg if reg is a floating point fsky. If
        reg is a string LR region, translates to appropriate fsky to use in
        modeling (i.e. fsky, not fsky_eff)"""

        if type(self.reg) is str:
            x = {'LR16':0.2, 'LR24':0.3, 'LR33':0.4, 'LR42':0.5, 'LR53':0.6, 
                 'LR63N':0.7, 'LR63S':0.7, 'LR63':0.7, 'LR72':0.8}
            self.fsky = x[self.reg]
        else:
            self.fsky = self.reg


    def prepmaps(self, type, rlz=1):
        """Load maps, apply any masking, output as nside=512"""

        if type in ['pysm','all']:
            print('loading PySM maps...')
            fn = self.get_map_filenames('pysm')
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)

            # Add LCDM
            mod = Model(fsky=self.fsky)
            dum, lcdm_alm = mod.genlcdmmap(fwhm=0, lensingB=True)
            fwhm = [4.99, 4.82] # 217,353
            map217 = hp.alm2map(lcdm_alm, self.nside, lmax=self.nside*2, fwhm=fwhm[0]*arcmin2rad)
            map353 = hp.alm2map(lcdm_alm, self.nside, lmax=self.nside*2, fwhm=fwhm[1]*arcmin2rad)

            maps[0] = maps[0] + map217
            maps[1] = maps[1] + map353
            maps[2] = maps[2] + map217
            maps[3] = maps[3] + map353
            maps[4] = maps[4] + map217
            maps[5] = maps[5] + map353

            maps = np.array(maps)
            maps = self.applypixwin(maps)

            self.s = maps

        if type == 'gaussian':
            print('generating Gaussian LCDM and dust maps')

            lensingB = True

            fwhm = [4.99, 4.82] # 217,353
            mod = Model(fsky=self.fsky)
            dum, lcdm_alm = mod.genlcdmmap(fwhm=0, lensingB=lensingB)
            dum, dust353_alm = mod.gengalmap(f=353, fwhm=0)
            fac = mod.freq_scale(217)

            alm_217 = lcdm_alm + dust353_alm*fac
            alm_353 = lcdm_alm + dust353_alm

            map217 = hp.alm2map(alm_217, self.nside, lmax=self.nside*2, fwhm=fwhm[0]*arcmin2rad)
            map353 = hp.alm2map(alm_353, self.nside, lmax=self.nside*2, fwhm=fwhm[1]*arcmin2rad)

            maps = []
            maps.append(map217)
            maps.append(map353)
            maps.append(map217)
            maps.append(map353)
            maps.append(map217)
            maps.append(map353)

            maps = np.array(maps)
            maps = self.applypixwin(maps)

            self.s = maps
        
        if type == 'db':
            """Constructing debiasing maps from sims"""
            mapsEnoB = []
            mapsBnoE = []
            for k,val in enumerate(self.s):
                alm = np.array(hp.map2alm(val, lmax=2*self.nside))

                almBnoE = dc(alm)
                almEnoB = dc(alm)

                almBnoE[1]=0
                almEnoB[2]=0

                mapsEnoB.append(hp.alm2map(tuple(almEnoB), self.nside, pol=True))
                mapsBnoE.append(hp.alm2map(tuple(almBnoE), self.nside, pol=True))

            self.EnoB = np.array(mapsEnoB)
            self.BnoE = np.array(mapsBnoE)


        if type in ['real','all']:
            print('loading real maps...')
            fn = self.get_map_filenames('real')
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.r = maps

        if type in ['realds']:
            print('loading real maps...')
            fn = self.get_map_filenames('realds')
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.r = maps

        if type in ['realhmds1', 'realhmds2', 'realhrds']:
            print('loading real maps...')
            fn = self.get_map_filenames(type)
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.r = maps

        if type in ['realhmdsavg']:
            print('loading real maps...')
            fn = self.get_map_filenames('realhmds1')
            maps1 = self.loadmaps(fn)
            maps1 = self.downgrade(maps1)
            fn = self.get_map_filenames('realhmds2')
            maps2 = self.loadmaps(fn)
            maps2 = self.downgrade(maps2)

            self.r = (maps1 + maps2)/2

        if type in ['noihmds1','noihmds2']:
            print('loading noise maps...')
            fn = self.get_map_filenames(type, rlz=rlz)
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.n = maps

        if type in ['noihmdsavg']:
            print('loading noise maps...')
            fn = self.get_map_filenames('noihmds1', rlz=rlz)
            maps1 = self.loadmaps(fn)
            maps1 = self.downgrade(maps1)
            fn = self.get_map_filenames('noihmds2', rlz=rlz)
            maps2 = self.loadmaps(fn)
            maps2 = self.downgrade(maps2)

            self.n = (maps1 + maps2)/2

        if 'qucov' in type:
            fn = self.get_map_filenames(type.replace('plusexcess',''), rlz=rlz)
            maps = self.loadmaps(fn)
            if 'plusexcess' in type:
                fn = self.get_map_filenames('excess_noise', rlz=rlz)
                maps2 = self.loadmaps(fn)
                maps = maps + maps2
            self.n = maps

        if type == 'qucovplusexcess_noise':
            print('loading noise maps...')
            fn = self.get_map_filenames('qucov_noise', rlz=rlz)
            maps1 = self.loadmaps(fn)
            fn = self.get_map_filenames('excess_noise', rlz=rlz)
            maps2 = self.loadmaps(fn)

            self.n = maps1 + maps2

        if type in ['mc_noise','all']:
            print('loading noise maps...')
            fn = self.get_map_filenames('mc_noise', rlz=rlz)
            maps = self.loadmaps(fn)            
            maps = self.downgrade(maps)
            self.n = maps
            self.rlz = rlz

        if type == 'mcplusexcess_noise':
            print('loading noise maps...')
            fn = self.get_map_filenames('mc_noise', rlz=rlz)
            maps1 = self.loadmaps(fn)            
            maps1 = self.downgrade(maps1)
            fn = self.get_map_filenames('excess_noise', rlz=rlz)
            maps2 = self.loadmaps(fn)

            self.n = maps1 + maps2
            self.rlz = rlz

        if type == 'hm217':
            print('loading intrafrequency hm split')
            fn = self.get_map_filenames('hm217')
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.hm217 = maps

        if type == 'hm353':
            print('loading intrafrequency hm split')
            fn = self.get_map_filenames('hm353')
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.hm353 = maps

        if type == 'ds217':
            print('loading intrafrequency ds split')
            fn = self.get_map_filenames('ds217')
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.ds217 = maps

        if type == 'ds353':
            print('loading intrafrequency hm split')
            fn = self.get_map_filenames('ds353')
            maps = self.loadmaps(fn)
            maps = self.downgrade(maps)
            self.ds353 = maps

        if type in ['sn','all']:
            print('adding s and n maps')
            self.sn = self.s + self.n


    def prepmasks(self, type):
        """Get point source mask and galaxy mask and downgrade as
        necessary. Point source mask has added complication that when
        downgrading, the edge pixels of a hole convolve with the surrouding
        ones, making them 0-1. Set these to zero"""

        if type in ['gal','all']:
            print 'loading LR mask...'
            gm = self.loadgalmask(self.reg)
            gm = self.downgrade(gm)
            self.gm = gm

        if type in ['nan','all']:
            print 'computing NaN mask'
            # Only use pixels that have a defined value in T, Q and U at all
            # frequencies in all splits in all types. S+n gets s, n, and sn
            # types 
            if hasattr(self,'n'):
                nsum = self.n.sum(0).sum(0) 
            else:
                nsum = 1

            if hasattr(self,'r'):
                rsum = self.r.sum(0).sum(0)
            else:
                rsum = 1

            nm = (np.isfinite(nsum)) & (np.isfinite(rsum))

            self.nm = nm.astype('float')


    def downgrade(self, maps):
        """Downgrade map nside if necessary"""

        # If input is a single map, put it in a list so we can loop over it
        if np.ndim(maps) == 1:
            wasnotlist = True
            maps = np.array([[maps]])
        else:
            wasnotlist = False
        
        # Do the downgrade if nside_out != nside_in
        nside_in = hp.npix2nside(maps[0][0].size)
        if nside_in != self.nside:
            print('downgrading from Nside {0} to {1}'.format(nside_in,self.nside))
            npix_out = hp.nside2npix(self.nside)
            sz = list(maps.shape)
            sz[2] = npix_out
            maps_out = np.zeros(sz)
            for k,val in enumerate(maps):
                val[~np.isfinite(val)] = udefval
                
                maps_out[k] = hp.pixelfunc.ud_grade(val, self.nside, pess=False)

        else:
            maps_out = dc(maps)

        if wasnotlist:
            maps_out = maps_out[0][0]

        # ud_grade is lame and sets bad pixels to the undefined value. Make
        # these nan
        maps_out[maps_out == udefval] = np.nan

        return maps_out


    def applypixwin(self, maps):
        """Apply the pixel window function to a set of maps"""

        pw = hp.sphtfunc.pixwin(self.nside)

        for k,val in enumerate(maps):
            alm = np.array(hp.map2alm(val))
            for j,val in enumerate(alm):
                alm[j] = hp.almxfl(alm[j], pw)
            alm = (alm[0], alm[1], alm[2])
            maps[k] = hp.alm2map(alm, self.nside)

        return maps

    def get_map_filenames(self, maptype, rlz=1):
        """Get map filenames, maptype = 'real', 'pysm', or 'noise' """
        
        fn = []

        if maptype == 'real':
            basedir = 'maps/real/'
            fn.append(basedir + 'HFI_SkyMap_217_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217_512dg_R2.02_halfmission-1.fits')
            fn.append(basedir + 'HFI_SkyMap_353_512dg_R2.02_halfmission-1.fits')
            fn.append(basedir + 'HFI_SkyMap_217_512dg_R2.02_halfmission-2.fits')
            fn.append(basedir + 'HFI_SkyMap_353_512dg_R2.02_halfmission-2.fits')

        if maptype == 'realds':
            basedir = 'maps/real/'
            fn.append(basedir + 'HFI_SkyMap_217_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_full.fits')

        if maptype == 'realhmds1':
            basedir = 'maps/real/'
            fn.append(basedir + 'HFI_SkyMap_217_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_halfmission-1.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_halfmission-1.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_halfmission-2.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_halfmission-2.fits')

        if maptype == 'realhmds2':
            basedir = 'maps/real/'
            fn.append(basedir + 'HFI_SkyMap_217_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_halfmission-2.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_halfmission-2.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_halfmission-1.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_halfmission-1.fits')

        if maptype == 'noihmds1':
            basedir = 'maps/qucov_noise/'
            fn.append(basedir + '217/ffp8_noise_217_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            basedir = 'maps/hmds/'
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_halfmission-1_noi{:05d}.fits'.format(rlz))
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_halfmission-1_noi{:05d}.fits'.format(rlz))
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_halfmission-2_noi{:05d}.fits'.format(rlz))
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_halfmission-2_noi{:05d}.fits'.format(rlz))

        if maptype == 'noihmds2':
            basedir = 'maps/qucov_noise/'
            fn.append(basedir + '217/ffp8_noise_217_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            basedir = 'maps/hmds/'
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_halfmission-2_noi{:05d}.fits'.format(rlz))
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_halfmission-2_noi{:05d}.fits'.format(rlz))
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_halfmission-1_noi{:05d}.fits'.format(rlz))
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_halfmission-1_noi{:05d}.fits'.format(rlz))

        if maptype == 'realhrds':
            basedir = 'maps/real/'
            fn.append(basedir + 'HFI_SkyMap_217_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_full-ringhalf-1.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_full-ringhalf-1.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_full-ringhalf-2.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_full-ringhalf-2.fits')
            
        if maptype == 'pysm':
            basedir = '../PySM/'
            fn.append(self.get_pysm_fname(217, self.comp, self.simprefix))
            fn.append(self.get_pysm_fname(353, self.comp, self.simprefix))
            fn.append(self.get_pysm_fname(217, self.comp, self.simprefix))
            fn.append(self.get_pysm_fname(353, self.comp, self.simprefix))
            fn.append(self.get_pysm_fname(217, self.comp, self.simprefix))
            fn.append(self.get_pysm_fname(353, self.comp, self.simprefix))

        if maptype == 'mc_noise':
            basedir = 'maps/mc_noise/'
            fn.append(basedir + '217/ffp8_noise_217_full_map_mc_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_full_map_mc_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '217/ffp8_noise_217_hm1_map_mc_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_hm1_map_mc_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '217/ffp8_noise_217_hm2_map_mc_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_hm2_map_mc_512dg_{:05d}.fits'.format(rlz))

        if maptype == 'excess_noise':
            basedir = 'maps/excess_noise/'
            fn.append(basedir + 'hm1_217_{:04d}.fits'.format(rlz))
            fn.append(basedir + 'hm2_353_{:04d}.fits'.format(rlz))
            fn.append(basedir + 'hm1_217_{:04d}.fits'.format(rlz))
            fn.append(basedir + 'hm1_353_{:04d}.fits'.format(rlz))
            fn.append(basedir + 'hm2_217_{:04d}.fits'.format(rlz))
            fn.append(basedir + 'hm2_353_{:04d}.fits'.format(rlz))

        if maptype == 'qucov_noise':
            basedir = 'maps/qucov_noise/'
            fn.append(basedir + '217/ffp8_noise_217_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '217/ffp8_noise_217_hm1_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_hm1_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '217/ffp8_noise_217_hm2_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_hm2_map_qucov_512dg_{:05d}.fits'.format(rlz))

        if maptype == 'qucovds_noise':
            basedir = 'maps/qucov_noise/'
            fn.append(basedir + '217/ffp8_noise_217_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_full_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '217/ffp8_noise_217_ds1_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_ds1_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '217/ffp8_noise_217_ds2_map_qucov_512dg_{:05d}.fits'.format(rlz))
            fn.append(basedir + '353/ffp8_noise_353_ds2_map_qucov_512dg_{:05d}.fits'.format(rlz))

        if maptype == 'ds217':
            basedir = 'maps/real/'
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_full-ringhalf-1.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_full-ringhalf-1.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds1_512dg_R2.02_full-ringhalf-2.fits')
            fn.append(basedir + 'HFI_SkyMap_217-ds2_512dg_R2.02_full-ringhalf-2.fits')

        if maptype == 'ds353':
            basedir = 'maps/real/'
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_full.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_full-ringhalf-1.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_full-ringhalf-1.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds1_512dg_R2.02_full-ringhalf-2.fits')
            fn.append(basedir + 'HFI_SkyMap_353-ds2_512dg_R2.02_full-ringhalf-2.fits')


        return fn
            

    def genqucov(self, type='real'):
        """Gen random noise realization from QU covariance maps"""
        
        # Get real map names
        fn = self.get_map_filenames(type)
        fnmc = self.get_map_filenames('mc_noise', rlz=0)

        maps = []
        # Load covariance maps
        for k,val in enumerate(fn):
            # Ordering is TT, QQ, QU, UU
            fn0 = val.replace('512dg','2048')
            covmap = hp.read_map(fn0, field=(4,7,8,9))
            
            # Generate random numbers 
            npix = covmap[0].size
            Q = np.random.randn(npix)
            U = np.random.randn(npix)

            # Generate correlated random numbers
            r = covmap[2]/np.sqrt(covmap[1]*covmap[3])
            U = r*Q + np.sqrt(1-r**2)*np.random.randn(npix)

            # Give both the correct variance
            Q = Q*np.sqrt(covmap[1])
            U = U*np.sqrt(covmap[3])
            
            # Generate uncorrelated T
            T = np.random.randn(npix)*np.sqrt(covmap[0])

            # Apply hm mask
            x = fnmc[k]
            x = x.replace('_512dg_','_')
            nm = hp.read_map(x,field=1)
            nm = np.isfinite(nm)
            T[~nm] = udefval
            Q[~nm] = udefval
            U[~nm] = udefval

            # Downgrade and append
            maps.append( self.downgrade(np.array([[T,Q,U]]) ))
            
        return np.squeeze(np.array(maps))
        


    def get_pysm_fname(self, f, comp, prefix='', nside=512, mapdir=None):
        """Get PySM filename for input frequency in GHz, and input list combination of
        components"""

        pysmpath=os.getenv('PYSMPATH')
        if mapdir is None:
            mapdir=pysmpath+'/Output/'

        # Massage components into correct order, which appear to be alphabetic
        cs=np.sort(comp)

        typestr = ''
        ncomp = len(comp)

        fn=prefix
        for k,val in enumerate(cs):
            fn=fn+'{0}_'.format(val)

        freqsuffix='{:.1f}_'.format(f).replace('.','p')

        fnout='{0}{1}{2}.fits'.format(fn,freqsuffix,np.str(np.round(nside)))

        return mapdir+fnout


    def loadmaps(self, fn):
        """Load healpix maps given array of filenames"""

        maps = []
        for k,val in enumerate(fn):
             maps.append(hp.fitsfunc.read_map(val, field=(0,1,2)))
        
        maps = np.array(maps)

        return maps
                    

    def loadgalmask(self, region):
        """Get galaxy map defined for PIPL LR regions (e.g. region='LR16') or
        fsky (e.g. region = 0.7)"""

        if type(region) is str:
            x = {'LR16':0, 'LR24':1, 'LR33':2, 'LR42':3, 'LR53':4, 
                 'LR63N':5, 'LR63S':6, 'LR63':7, 'LR72':8}
            hmap = hp.fitsfunc.read_map('maps/masks/COM_Mask_Dust-diffuse-and-ps-PIP-L_0512_R2.00.fits',
                                        field=x[region])
        else:
            x={0.2:0, 0.4:1, 0.6:2, 0.7:3, 0.8:4, 0.9:5, 0.97:6, 0.99:7}
            hmap = hp.fitsfunc.read_map('maps/masks/HFI_Mask_GalPlane-apo5_2048_R2.00.fits',
                                        field=x[region])

        return hmap


    def getmask(self):
        return self.gm*self.nm


    def writemaps(self, maptype):
        """Write out the downgraded maps."""
        print('writing maps')
        fn = self.get_map_filenames(maptype, self.rlz)
        if maptype == 'noise':
            field = 'n'
        if maptype == 'real':
            field = 'r'

        maps = getattr(self, field)
        for k,val in enumerate(fn):
            if field == 'n':
                valout = str.replace(val, '_map_mc', '_map_mc_{0}dg'.format(self.nside))
            if field == 'r':
                valout = str.replace(val, '2048', '{0}dg'.format(self.nside))
            hp.write_map(valout, maps[k])



###################
###################

class Spec(object):
    """Compute spectra for map object"""

    def __init__(self, maps):

        # The maps field will get removed later
        self.maps = maps
        
        # Keep these around for reference
        self.reg = maps.reg
        self.fsky = maps.fsky


    def getspec(self, field, append=False, estimator='pspice'):
        """If append=True, add a new realization to existing
        estimator = 'pspice', 'xpol'
        """
        
        print('getting spec {0}'.format(field))

        # Get mask
        self.mask = self.maps.getmask()

        if estimator == 'pspice':
            if type(field) is not list:
                s0 = np.array(self.getautocross_pspice(getattr(self.maps,field), self.mask))
            else:
                s0 = np.array(self.getautocross_pspice(getattr(self.maps,field[0]) +
                                                       getattr(self.maps,field[1]),
                                                       self.mask))
        if estimator == 'xpol':
            if type(field) is not list:
                s0 = np.array(self.getautocross_xpol(getattr(self.maps,field), self.mask))
            else:
                s0 = np.array(self.getautocross_xpol(getattr(self.maps,field[0]) +
                                                     getattr(self.maps,field[1]),
                                                     self.mask))

        if estimator == 'hp':
            if type(field) is not list:
                s0 = np.array(self.getautocross(getattr(self.maps,field), self.mask))
            else:
                s0 = np.array(self.getautocross(getattr(self.maps,field[0]) +
                                                getattr(self.maps,field[1]),
                                                self.mask))

        # Append or not
        if not append or not hasattr(self, field):
            field = string.replace(string.join(field),' ','')
            setattr(self, field, s0)
        else:
            x = getattr(self, field)
            if type(x) is not list:
                setattr(self, field, [x])
            x = getattr(self, field)
            x.append(s0)
        
        return
    

    def getautocross_pspice(self, maps, mask):
        """Calculate auto and cross of maps*mask:
           auto1  = maps[2] x maps[4]
           auto2  = maps[3] x maps[5]
           cross  = (maps[2] x maps[3] + maps[2] x maps[5] + maps[4] x maps[3] + maps[4] x maps[5])/4
           crossf = maps[0] x maps[1]
           """
        
        # Zero maps where mask=0 to avoid possible NaNs in the masked region.
        x = np.ones(maps.shape)*mask
        maps[x==0]=0

        # Write necessary files
        rands = randstring(6)
        tempdir = 'tempmaps'
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        # Write mask
        maskfn = '{0}/mask_{1}.fits'.format(tempdir,rands)
        hp.write_map(maskfn, mask)
        fsky = np.nanmean(mask)

        # Write maps
        mapfn = []
        for k,val in enumerate(maps):
            mapfn.append('{0}/map_{1}_{2}.fits'.format(tempdir,k,rands))
            hp.write_map(mapfn[k], val)

        # Get C_ls
        auto1fn = '{0}/auto1_{1}.fits'.format(tempdir, rands)
        auto2fn = '{0}/auto2_{1}.fits'.format(tempdir, rands)
        crossAfn = '{0}/crossA_{1}.fits'.format(tempdir, rands)
        crossBfn = '{0}/crossB_{1}.fits'.format(tempdir, rands)
        crossCfn = '{0}/crossC_{1}.fits'.format(tempdir, rands)
        crossDfn = '{0}/crossD_{1}.fits'.format(tempdir, rands)
        crossffn = '{0}/crossf_{1}.fits'.format(tempdir, rands)

        lmax = 720

        self.callispice(mapfn[2], mapfn[4], maskfn, auto1fn, fsky, lmax)
        self.callispice(mapfn[3], mapfn[5], maskfn, auto2fn, fsky, lmax)
        self.callispice(mapfn[2], mapfn[3], maskfn, crossAfn, fsky, lmax)
        self.callispice(mapfn[2], mapfn[5], maskfn, crossBfn, fsky, lmax)
        self.callispice(mapfn[4], mapfn[3], maskfn, crossCfn, fsky, lmax)
        self.callispice(mapfn[4], mapfn[5], maskfn, crossDfn, fsky, lmax)
        self.callispice(mapfn[0], mapfn[1], maskfn, crossffn, fsky, lmax)

        auto1  = np.array(hp.read_cl(auto1fn)[0:3])
        auto2  = np.array(hp.read_cl(auto2fn)[0:3])
        crossA = np.array(hp.read_cl(crossAfn)[0:3])
        crossB = np.array(hp.read_cl(crossBfn)[0:3])
        crossC = np.array(hp.read_cl(crossCfn)[0:3])
        crossD = np.array(hp.read_cl(crossDfn)[0:3])
        crossf = np.array(hp.read_cl(crossffn)[0:3])

        cross = (crossA + crossB + crossC + crossD)/4.0

        os.remove(auto1fn)
        os.remove(auto2fn)
        os.remove(crossAfn)
        os.remove(crossBfn)
        os.remove(crossCfn)
        os.remove(crossDfn)
        os.remove(crossffn)
        os.remove(maskfn)
        for k,val in enumerate(mapfn):
            os.remove(val)

        # Set ell, return spectra. This is dumb.
        self.l = np.arange(auto1.shape[1])

        return auto1, auto2, cross, crossf


    def getautocross_xpol(self, maps, mask):
        """Calculate auto and cross of maps*mask:
           auto1  = maps[2] x maps[4]
           auto2  = maps[3] x maps[5]
           cross  = (maps[2] x maps[3] + maps[2] x maps[5] + maps[4] x maps[3] + maps[4] x maps[5])/4
           crossf = maps[0] x maps[1]
           """
        
        # Zero maps where mask=0 to avoid possible NaNs in the masked region.
        x = np.ones(maps.shape)*mask
        maps[x==0]=0

        # Write necessary files
        rands = randstring(6)
        tempdir = 'tempmaps'
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        # Write mask
        maskfn = '{0}/mask_{1}.fits'.format(tempdir,rands)
        hp.write_map(maskfn, mask)
        fsky = np.nanmean(mask)

        # Write maps
        mapfn = []
        for k,val in enumerate(maps):
            mapfn.append('{0}/map_{1}_{2}.fits'.format(tempdir,k,rands))
            hp.write_map(mapfn[k], val)

        lmax = 700
        be = np.arange(0, lmax+1, 10)

        l, auto1, dum = xpol.getXcorr(512, mapfn[2], mapfn[4], maskfn, be)
        l, auto2, dum = xpol.getXcorr(512, mapfn[3], mapfn[5], maskfn, be)
        l, crossA, dum = xpol.getXcorr(512, mapfn[2], mapfn[3], maskfn, be)
        l, crossB, dum = xpol.getXcorr(512, mapfn[2], mapfn[5], maskfn, be)
        l, crossC, dum = xpol.getXcorr(512, mapfn[4], mapfn[3], maskfn, be)
        l, crossD, dum = xpol.getXcorr(512, mapfn[4], mapfn[5], maskfn, be)
        l, crossf, dum = xpol.getXcorr(512, mapfn[0], mapfn[1], maskfn, be)

        cross = (crossA + crossB + crossC + crossD)/4.0

        os.remove(maskfn)
        for k,val in enumerate(mapfn):
            os.remove(val)

        # Set l, return spectra. This is dumb.
        self.l = l

        # Return C_l
        fac = 2*np.pi/(l*(l+1))

        return fac*auto1, fac*auto2, fac*cross, fac*crossf


    def callispice(self, map1, map2, mask, clout, fsky, lmax):
        """Call PolSpice, wants filenames"""

        # apodizesigma and thetamax are supposed to scale with fsky (from
        # PolSpice documentation). Turns out that apodizetype=1 is absolutley
        # critical.
        
        # Tolerance must be lower for small fsky, too
        if fsky >= 0.5:
            tol = 1e-6
        if fsky<0.5:
            tol = 5e-8

        th = round(np.interp(fsky, [0.01,0.5], [20,180]))
        th = np.min([th,180])
        ispice(map1, clout, mapfile2=map2, weightfile1=mask, weightfile2=mask,
               polarization='YES', decouple='YES', tolerance=tol, 
               subav='YES', subdipole='YES', apodizesigma=th, thetamax=th,
               nlmax=lmax, apodizetype=1)
        return
                                    

    def getautocross(self, maps, mask):
        """Calculate auto and cross of maps*mask:
           auto1  = maps[2] x maps[4]
           auto2  = maps[3] x maps[5]
           cross  = (maps[2] x maps[3] + maps[2] x maps[5] + maps[4] x maps[3] + maps[4] x maps[5])/4
           crossf = maps[0] x maps[1]
           """
        
        # Apply mask
        x = self.applymask(maps, mask)
        
        # Normalization factor
        fac = 1/np.nanmean(mask**2)

        # Get alms
        alm = self.getalms(x)

        # [T,E,B]
        auto1 = [[],[],[]]
        auto2 = [[],[],[]]
        cross = [[],[],[]]
        crossf = [[],[],[]]

        for k,val in enumerate(cross):
            auto1[k] = hp.alm2cl(alm[2][k],alm[4][k])
            auto2[k] = hp.alm2cl(alm[3][k],alm[5][k])
            cross[k] = (hp.alm2cl(alm[2][k],alm[3][k]) + 
                        hp.alm2cl(alm[2][k],alm[5][k]) + 
                        hp.alm2cl(alm[4][k],alm[3][k]) + 
                        hp.alm2cl(alm[4][k],alm[5][k]))/4.0
            crossf[k] = hp.alm2cl(alm[0][k],alm[1][k])

        self.l = np.arange(auto1[0].size)

        return fac*np.array(auto1), fac*np.array(auto2), fac*np.array(cross), fac*np.array(crossf)
        

    def getalms(self, maps):
        """Calculate alms"""
        lmax = self.maps.nside * 2
        alm = []
        for k,val in enumerate(maps):
            alm.append(hp.map2alm(val))
        return alm


    def applymask(self, maps_in, mask):
        """Set maps to zero where mask = 0 (avoids 0*nan = nan) and then
        multiply by mask (to apodize)"""
        # Expand mask to maps size
        x = np.ones(maps_in.shape)*mask

        # Multiply
        maps_out = maps_in*x
        
        # Set to zero
        maps_out[x==0]=0

        return maps_out
        


###################
###################

class Calc(object):
    """Calculate decorrelation parameter for input Spec object"""

    def __init__(self, spec, bintype='planck', lmin=10, lmax=1000, nbin=100,
                 full=False, dodebias=False, binwhat='spec', Rtype='R',doall=True):

        if type(spec) is str:
            # It's a pickle file, load it
            f = open(spec,'rb')
            self.spec = cP.load(f)
            f.close()
        else:
            # It's already loaded
            self.spec = dc(spec)
        
        self.bintype = bintype
        self.binwhat = binwhat
        self.nbin = nbin
        self.lmin = lmin
        self.lmax = lmax
        self.full = full
        self.dodebias = dodebias
        self.Rtype = Rtype

        if doall:
            self.doall()

        return

    def doall(self):
        """Only works with all s, n, sn and r components."""
        # Add a dimension if only 1 realization
        self.addspecdim()
        if self.dodebias:
            self.debias()
        self.binspec()
        self.getR()

        # Mean of s,n, and sn over realizations
        self.Smean = np.nanmean(self.S, 0)
        self.Nmean = np.nanmean(self.N, 0)
        self.SNmean = np.nanmean(self.SN, 0)
        self.Smedian = np.nanmedian(self.S, axis=0)
        self.Nmedian = np.nanmedian(self.N, axis=0)
        self.SNmedian = np.nanmedian(self.SN, axis=0)

        # Error is std of S+N sims over realizations
        #self.err = np.nanstd(self.SN, 0)
        
        # Error is median absolute deviation
        self.err = np.nanmedian(np.abs(self.SN - self.SNmedian), axis=0)

        self.nrlz = self.SN.shape[0]
        if self.nrlz > 1:
            self.getPTE()

        self.getRdist()
        self.getRlike()

    def addspecdim(self):
        """Add a dimension to s, n, and sn if only 1 realization"""
        if self.spec.s.ndim == 3:
            self.spec.s = self.spec.s.reshape((1,) + self.spec.s.shape)
        if self.spec.n.ndim == 3:
            self.spec.n = self.spec.n.reshape((1,) + self.spec.n.shape)
        if self.spec.sn.ndim == 3:
            self.spec.sn = self.spec.sn.reshape((1,) + self.spec.sn.shape)



    def binspec(self, type_in='all'):
        
        self.getbins()

        if type_in == 'all':
            type = ['r','s','n','sn']
        else:
            type = [type_in]

        for dum,val in enumerate(type):
            x = getattr(self.spec,val)
            if len(x.shape) == 3:
                # Only 1 "realization" #
                setattr(self, val, self.binspec_sub(x))
            else:
                setattr(self, val, np.array([self.binspec_sub(x[k]) for k in range(len(x))]) )

        return

    def getR(self, type_in='all'):

        if type_in == 'all':
            type = ['R','S','N','SN']
        else:
            type = [type_in]

        for dum,val in enumerate(type):
            x = getattr(self, val.lower())
            if len(x.shape) == 3:
                # Only 1 "realization"
                setattr(self, val, self.calcR(x))
            else:
                setattr(self, val, np.array([self.calcR(x[k]) for k in range(len(x))]) )
                
        return


    def debias(self):
        """Debias noise bias from spectra"""

        # Average over realizations
        self.spec.db = np.zeros(self.spec.r.shape)

        if type(self.dodebias) is str:
            # Load file
            f = open(self.dodebias)
            s = cP.load(f)
            self.spec.db[:,1,:] = s.n.mean(0)[:,1,:]
            self.spec.db[:,2,:] = s.n.mean(0)[:,2,:]
        else:
            self.spec.db[:,1,:] = self.spec.n.mean(0)[:,1,:]
            self.spec.db[:,2,:] = self.spec.n.mean(0)[:,2,:]

        self.spec.r = self.spec.r - self.spec.db
        
        if type(self.dodebias) is not str:
            for k in range(self.spec.s.shape[0]):
                self.spec.sn[k] = self.spec.sn[k] - self.spec.db


    def getbins(self):
        """Define bins, currently Planck paper's"""
        if self.bintype == 'planck':
            self.be = np.array([50,160,320,500,700])
            self.nbin = 4

        if self.bintype == 'bk':
            self.be = np.array([20,55,90,125,160,195,230,265,300,335])
            self.nbin = 9

        if self.bintype == 'log':
            self.be = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nbin+1)
            
        if self.bintype == 'lin':
            self.be = np.linspace(self.lmin, self.lmax, self.nbin+1)

        self.bc = (self.be[0:-1] + self.be[1:])/2

        return


    def binspec_sub(self, spec):
        """Bin spectra"""

        l = self.spec.l
        be = self.be
        sz = spec.shape
        fac = l*(l+1)/(2*np.pi)

        if self.binwhat == 'spec':
            # Bin spectra
            specbin = np.zeros((sz[0],sz[1],self.bc.size))
            for k in range(0,sz[0]):
                for j in range(0,sz[1]):
                    for m in range(0,self.bc.size):
                        specbin[k][j][m] = np.mean( (fac*spec[k][j])[(l>=be[m]) & (l<be[m+1])] )

        elif self.binwhat == 'R':
            # Bin R
            specbin = np.zeros((sz[0],self.bc.size))
            for k in range(0,sz[0]):
                for m in range(0,self.bc.size):
                    specbin[k][m] = np.nanmean( (fac*spec[k])[(l>=be[m]) & (l<be[m+1])] )

        return specbin


    def calcR(self, specbin):
        """Calculate decorrelation ratio. If self.full=True, use full mission for
        cross spectrum, otherwise use mean of half mission crosses."""
        if self.binwhat == 'spec':
            R = np.zeros((3,self.bc.size))
        else:
            R = np.zeros((3,self.spec.l.size))

        if self.full:
            crossind = 3
        else:
            crossind = 2

        for k in range(3):
            if self.Rtype == 'R':
                R[k] = specbin[crossind][k]/np.sqrt(specbin[0][k]*specbin[1][k])
            if self.Rtype == 'Rsquared':
                R[k] = specbin[crossind][k]**2/(specbin[0][k]*specbin[1][k])
        return R


    def getPTE(self):
        """Get PTE values from sims"""
        # Bin by bin
        sz = self.R.shape
        self.PTE = np.zeros(sz)
        self.PTEall = np.zeros(sz[0])
        for k in range(sz[0]):
            for j in range(sz[1]):
                ngood = np.where(np.isfinite(self.SN[:,k,j]))[0].size
                if ngood>0:
                    self.PTE[k,j] = np.size(np.where(self.SN[:,k,j] < self.R[k,j])[0])*1.0/ngood
                else:
                    self.PTE[k,j] = np.nan
        self.PTE[~np.isfinite(self.R)] = np.nan

        # Combined PTE
        # Impose R nanmask on SN
        SN = self.SN
        R = self.R
        nanmask = np.ones(R.shape)
        nanmask[~np.isfinite(R)]=np.nan
        SN = SN*nanmask

        doind = (self.bc >=50) & (self.bc<=700)
        Rcomb = np.nanmean((R/self.err**2)[:,doind]**2, 1)
        SNcomb = np.nanmean((SN/self.err**2)[:,:,doind]**2, 2)
        for k in range(sz[0]):
            ngood = np.where(np.isfinite(SNcomb[:,k]))[0].size
            self.PTEall[k] = np.size( np.where(SNcomb[:,k] < Rcomb[k])[0] )*1.0/ngood



    def getmode(self):
        """Calculate most likely value of R distribution."""

        # First fit all the SN R distributions
        self.fitRdist()

        # Look for max of distribution over these values
        x = np.linspace(0,2,10000)
        sz = self.SN.shape

        self.mode = np.zeros((sz[1],sz[2]))

        for k in [1,2]:
            for j in range(sz[2]):
                # Get skew normal dist
                f = self.skewnorm(x, *self.p[:,k,j])
                mx = np.nanmax(f)
                try:
                    self.mode[k,j] = x[np.where(f==mx)]
                except:
                    self.mode[k,j] = np.nan


    def fitRdist(self):
        """Fit a skewnorm distribution to the SN ratios and get parameters"""
        nparams = 4
        sz = np.array(self.SN.shape)
        sz[0] = nparams

        self.p = np.zeros(sz)

        for k in [1,2]: # Only do E/B
            for j in range(sz[2]):
                self.p[:, k, j] = self.getskewnorm(self.SN[:,k,j])


    def getskewnorm(self, x, nbin=10):
        """Histogram and fit offset to collection of x"""

        # Get hist
        x = getfinite(x)

        # Get range
        mn = np.max([-5, np.min(x)])
        mx = np.min([20, np.max(x)])

        n, be = np.histogram(x, bins=nbin, range=(mn,mx))
        bc = (be[0:-1] + be[1:])/2
        
        # Initial guesses
        sigma0 = x.std()
        mu0 = x.mean()
        alpha0 = skew(x)
        a0 = np.max(n)

        p0 = np.array([sigma0, mu0, alpha0, a0])

        try:
            p, pcov = curve_fit(self.skewnorm, bc, n, p0=p0)
        except:
            p = p0; p[:] = np.nan

        return p


    def skewnorm(self, x, sigmag, mu, alpha, a):
        """Skew normal distribution"""
        normpdf = np.exp(-(x-mu)**2 / (2*sigmag**2))
        normcdf = 1+sp.erf((alpha*((x-mu)/sigmag))/(np.sqrt(2)))
        return a*normpdf*normcdf

    def getRdist(self):
        """Get max likelihood distribution for R centered on observed value"""
        mu = np.nanmean(self.s,0)

        R = self.R

        self.up68 = np.zeros(R.shape)
        self.up95 = np.zeros(R.shape)
        self.down68 = np.zeros(R.shape)
        self.down95 = np.zeros(R.shape)

        if self.full:
            crossind = 3
        else:
            crossind = 2

        mod = Model(fsky=self.spec.fsky, be=self.be)
        
        for k in [1,2]:
            for j in range(R.shape[1]):

                auto1 = self.sn[:,0,k,j]
                auto2 = self.sn[:,1,k,j]
                cross = self.sn[:,crossind,k,j]

                # Shift cross up or down
                if k==1:
                    Rmod = mod.RE[j]
                else:
                    Rmod = mod.RB[j]
                Rnew = R[k,j]
                dR = (Rmod - Rnew)
                dc = dR*np.sqrt(mu[0,k,j]*mu[1,k,j])

                x = (self.sn[:,3,k,j]-dc) / np.sqrt(self.sn[:,0,k,j]*self.sn[:,1,k,j])

                x = x[np.isfinite(x)]
                if len(x) > 20:
                    self.up68[k,j] = np.percentile(x, 50 + 68./2)
                    self.up95[k,j] = np.percentile(x, 50 + 95./2)
                    self.down68[k,j] = np.percentile(x, 50 - 68./2)
                    self.down95[k,j] = np.percentile(x, 50 - 95./2)
                else:
                    self.up68[k,j] = np.nan
                    self.up95[k,j] = np.nan
                    self.down68[k,j] = np.nan
                    self.down95[k,j] = np.nan


    
    def getRlike(self):
        """Get likelihood of observed value for different trial "true" R
        values, computed as fraction of shifted sim distributions that have 
        Rsim > Robs"""
        
        mu = np.nanmean(self.s,0)

        R = self.R
        sz = R.shape

        self.Rtrial = np.hstack( (np.linspace(0,.89,90),np.linspace(0.9,1,41)) )
        ntrial = self.Rtrial.size

        self.Rlike = np.zeros((ntrial,sz[0],sz[1]))

        if self.full:
            crossind = 3
        else:
            crossind = 2

        mod = Model(fsky=self.spec.fsky, be=self.be)
        
        for k in [1,2]:
            for j in range(R.shape[1]):

                auto1 = self.sn[:,0,k,j]
                auto2 = self.sn[:,1,k,j]
                cross = self.sn[:,crossind,k,j]

                # Shift cross down
                if k==1:
                    Rmod = mod.RE[j]
                else:
                    Rmod = mod.RB[j]

                for l,val in enumerate(self.Rtrial):
                    Rnew = val
                    dR = (Rmod - Rnew)
                    dc = dR*np.sqrt(mu[0,k,j]*mu[1,k,j])

                    x = (self.sn[:,3,k,j]-dc) / np.sqrt(self.sn[:,0,k,j]*self.sn[:,1,k,j])

                    x = x[np.isfinite(x)]
                    ngood = len(x)
                    if ngood > 20:
                        self.Rlike[l,k,j] = (np.where(x > R[k,j])[0]).size*1.0 / ngood
                    else:
                        self.Rlike[l,k,j] = np.nan



###################
###################

class Model(object):

    def __init__(self, fsky=0.7, l=None, be=None):
        """Model of R for given fsky"""
        if l is None:
            l = np.arange(1000)

        self.l = l
        self.fsky = fsky
        self.be = be

        RB,ll = self.getR(spec=2)
        RE,ll = self.getR(spec=1)

        if self.be is None:
            self.RB = np.interp(self.l, ll, RB)
            self.RE = np.interp(self.l, ll, RE)
        else:
            self.RB = RB
            self.RE = RE


    def genlcdmmap(self, fwhm=4.818, nside=512, lensingB=True):
        """Make a simulated LCDM TT+EE + diagonal lensing BB sky
        fwhm = beam FWHM in arcmin (default for Planck 353)"""

        # Get theory cl's
        cl,nm = self.readcambfits('camb_planck2013_r0_lensing.fits')

        # uK_cmb^2 -> K_cmb^2
        cl = cl*1e-12

        if not lensingB:
            # Zero lensing BB
            cl[:,2] = 0

        # TT, TE, EE, BB
        x = (cl[:,0],cl[:,3],cl[:,1],cl[:,2])

        # Make map 
        map,alm  = hp.synfast(x, nside=nside, alm=True, pol=True,
                          fwhm=(fwhm/60.0)*np.pi/180, lmax=2*nside, new=False)
        
        return np.asarray(map), np.asarray(alm)


    def gengalmap(self, f=353, fwhm=4.818, nside=512):
        """Simulate a full sky dust map for defined fsky of given Nside"""

        # Dummy cl's
        cl,nm = self.readcambfits('camb_planck2013_r0_lensing.fits')

        # Get dust Cl's
        l = np.arange(2001)
        clE = self.getdustcl(l, f, self.fsky, spec=1)*1e-12 #EE and uK^2->K^2
        clB = self.getdustcl(l, f, self.fsky, spec=2)*1e-12 #EE and uK^2->K^2
        clT = clE/.05**2
        clTE = np.zeros(clT.shape)

        # TT, TE, EE, BB
        x = (clT,clTE,clE,clB)

        # Make map 
        map,alm = hp.synfast(x, nside=nside, alm=True, pol=True,
                         fwhm=(fwhm/60.0)*np.pi/180, lmax=2*nside)


        return np.asarray(map), np.asarray(alm)

    def readcambfits(self, fname):
        """Read in a CAMB generated fits file and return a numpy array of the table
        values. Returns numpy array of N_l x N_fields and a string array of field
        names. Returns C_l's in uK^2.""" 

        h=fits.open(fname)
        d=h[1].data
        nm=d.names

        nl=d[nm[0]].size
        nfields=np.size(nm)

        cl=np.zeros([nl,nfields])

        for k,val in enumerate(nm):
            cl[:,k]=d[val]

        # Convert to uK^2
        cl = cl*1e12

        return cl,nm


    def getR(self, fsky=None, spec=2, be=None, Robs=None, syst=None):
        """fsky = sky fraction
           spec = 0,1,2 for T,E,B. TT does not currently work because the dust power
                  law needs to be defined. BICEP field corresponds to fsky=.135
           be = bin edges, if not None
           delta = dust only decorrelation param with no CMB, 1 by default for
                   not decorrelation
           Robs = if defined, will output the DUST ONLY decorrelation R for a
                   given observed R, which includes CMB and dust. Robs must be
                   size be-1
           syst = 3 x N x l array of systematics c_l in uK^2 (not d_l) to be added to the
                  217, 353, and cross spectra, respectively. First dim is 217,
                  353, cross, second dim is TETetc., third dim is ell
        """
        
        if fsky is None:
            fsky = self.fsky

        if be is None:
            be = self.be

        # Load lensing C_l in uK_cmb^2
        cl_l,nm = self.readcambfits('camb_planck2013_r0_lensing.fits');
        cl_l = cl_l[:,spec] # TT, EE, orBB
        l = np.arange(cl_l.size)

        # Get dust spectrum D_l in uK_cmb^2 at 353 and 217 GHz
        fac = l*(l+1)/(2*np.pi)
        dl_353 = self.getdustcl(l,353,fsky,spec)*fac
        dl_217 = self.getdustcl(l,217,fsky,spec)*fac    
        dl_cmb = cl_l*fac

        if be is not None:
            sz = be.size-1
            dl_cmb_bin = np.zeros(sz)
            dl_353_bin = np.zeros(sz)
            dl_217_bin = np.zeros(sz)

            for k in range(sz):
                ind = (l>=be[k]) & (l<be[k+1])
                dl_cmb_bin[k] = np.mean(dl_cmb[ind])
                dl_353_bin[k] = np.mean(dl_353[ind])
                dl_217_bin[k] = np.mean(dl_217[ind])

            dl_cmb = dl_cmb_bin
            dl_353 = dl_353_bin
            dl_217 = dl_217_bin
            
            l = (be[0:-1] + be[1:])/2.

        z = np.zeros( (3, l.size) )
        if syst is not None:
            # Expand systematics to size of spectra
            lmax = syst[0,0,:].size-1
            # Cl -> Dl
            ll = np.arange(lmax+1)
            fac = ll*(ll+1)/(2*np.pi)
            for j in range(z.shape[0]):
                z[j,0:(lmax+1)] = np.interp(l[0:(lmax+1)], ll, syst[j,spec,:]*fac)

        denom = np.sqrt((dl_353 + dl_cmb + z[1])*(dl_217 + dl_cmb + z[0]))
        cross = np.sqrt(dl_353)*np.sqrt(dl_217) + z[2]

        if Robs is None:
            # Compute R = <353 x 217> / sqrt(<353x353><217x217>)
            return (cross + dl_cmb)/denom, l
        else:
            # R_dust
            return (Robs*denom - dl_cmb)/cross, l



    def readcambfits(self, fname):
        """Read in a CAMB generated fits file and return a numpy array of the table
        values. Returns numpy array of N_l x N_fields and a string array of field
        names. Returns C_l's in uK^2.""" 

        h=fits.open(fname)
        d=h[1].data
        nm=d.names

        nl=d[nm[0]].size
        nfields=np.size(nm)

        cl=np.zeros([nl,nfields])

        for k,val in enumerate(nm):
            cl[:,k]=d[val]

        # Convert to uK^2
        cl = cl*1e12

        return cl,nm


    def getdustcl(self, l, f, fsky, spec):
        """Kludge up a BB dust spectrum as a function of l at frequency f (in GHz) and
        fsky"""

        # Dust D_l amplitude
        A = self.fsky2A(fsky)
        if spec==1:
            A_353 = A[0] # A_EE
        else:
            A_353 = A[1] # A_BB

        # Fiducial l and freq
        lfid = 80.
        ffid = 353.

        # Construct D_l
        # From PIP XXX, spectrum is a power law in ell with -2.42 slope.
        dl = A_353*(l/lfid)**(-2.42 + 2)

        # D_l -> C_l
        cl = 2*np.pi*dl/(l*(l+1))

        # Get dust scale factor
        sf = self.freq_scale(f,ffid)

        cl = cl * sf**2 

        cl[l==0]=0;

        return cl


    def freq_scale(self, f, ffid=353.0, beta=1.59, Tdust=19.6, nside=512, dbeta=None):
        """Conversion factor to scale dust map at frequency ffid (GHz) to f
        (GHz)."""

        # Make sure this works if f is an integer
        f = np.float(f)

        if dbeta is not None:
            npix = hp.nside2npix(nside)
            betarand = np.random.randn(npix)*dbeta
            betamap = betarand + beta
        else:
            betamap = beta

        # Conversion factor for graybody
        fac = planck.planck(f*1e9,Tdust)/planck.planck(ffid*1e9,Tdust) * (f/ffid)**betamap
        fac = fac.value

        # Conversion factor for thermodynamic temperature
        h = 6.626e-34
        kB = 1.381e-23
        Tcmb = 2.72548
        cf1 = f**4 * np.exp(h*f*1e9 / (kB*Tcmb)) / (np.exp(h*f*1e9 / (kB*Tcmb)) - 1)**2
        cf2 = ffid**4 * np.exp(h*ffid*1e9 / (kB*Tcmb)) / (np.exp(h*ffid*1e9 / (kB*Tcmb)) - 1)**2
        cf = cf1/cf2

        return fac/cf


    def fsky2A(self, fsky):
        """Interpolate A_EE and A_EE/A_BB as a function of fsky from Planck XXX
        table and return A_EE and A_BB"""

        fsky_i = np.array([0, .3, .4, .5, .6, .7, .8])
        A_EE_i = np.array([0, 37.5, 51.0 , 78.6, 124.2, 197.1, 328.0])
        rat_i =  np.array([0.49, 0.49, 0.48, 0.53, 0.54, 0.53, 0.53])

        A_EE = np.interp(fsky, fsky_i, A_EE_i)
        rat =  np.interp(fsky, fsky_i, rat_i)
        
        A_BB = A_EE * rat

        return A_EE, A_BB




