import numpy as np
import sys
sys.path.append('/astro/u/csheehy/python/PolSpice_v03-03-02/src')
import os
import healpy as hp
from ispice import ispice
import string
import random

def randstring(size=4):
    """Generate random string of size size"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(size))

def ispice_wrap(map1, map2, mask=None, lmax=720):
    """Call PolSpice, wants loaded in healpix maps"""
    
    # apodizesigma and thetamax are supposed to scale with fsky (from
    # PolSpice documentation). Turns out that apodizetype=1 is absolutley
    # critical.
    map1 = np.array(map1)
    map2 = np.array(map2)
    
    # Zero maps where mask=0 to avoid possible NaNs in the masked region.
    ind = mask==0
    for k in range(3):
        map1[k,ind] = 0
        map2[k,ind] = 0
    
    # Write necessary files
    rands = randstring(6)
    tempdir = 'tempmaps'
    if not os.path.exists(tempdir):
        os.mkdir(tempdir)

    # Write mask
    maskfn = '{0}/mask_{1}.fits'.format(tempdir,rands)
    hp.write_map(maskfn, mask)
    
    # Write maps
    map1fn = '{0}/map1_{1}.fits'.format(tempdir,rands)
    map2fn = '{0}/map2_{1}.fits'.format(tempdir,rands)
    hp.write_map(map1fn, map1)
    hp.write_map(map2fn, map2)

    # Tolerance must be lower for small fsky, too
    fsky = np.nanmean(mask)
    if fsky >= 0.5:
        tol = 1e-6
    if fsky<0.5:
        tol = 5e-8

    # Apodization (see pspice documentation)
    th = round(np.interp(fsky, [0.01,0.5], [20,180]))
    th = np.min([th,180])

    # Output C_l filename
    clout = '{0}/cl_{1}.fits'.format(tempdir,rands)

    # Call Pspice
    ispice(map1fn, clout, mapfile2=map2fn, weightfile1=maskfn, weightfile2=maskfn,
           polarization='YES', decouple='YES', tolerance=tol, 
           subav='YES', subdipole='YES', apodizesigma=th, thetamax=th,
           nlmax=lmax, apodizetype=1)

    # Get C_ls
    cl  = np.array(hp.read_cl(clout))

    # Remove files
    os.remove(map1fn)
    os.remove(map2fn)
    os.remove(maskfn)

    return cl

               
