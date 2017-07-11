import healpy as hp
import numpy as np
import os
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", dest="f", type="int", default=217)
parser.add_option("-r", dest="rlz", type="int", default=100)
(o, args) = parser.parse_args()

def get_mc_fnames(f, rlz):
    dir = 'maps/mc_noise/{:03d}/'.format(f)
    if f in [30, 44, 70]:
        hm = ['full', 'y13', 'y24']
    else:
        hm = ['full', 'hm1', 'hm2']

    fn = []
    
    for k,val in enumerate(hm):
        fn.append(dir + 'ffp8_noise_{:03d}_{:}_map_mc_{:05d}.fits'.format(f, val, rlz))

    return fn

def get_real_fnames(f):
    dir = 'maps/real/'
    if f in [30, 44, 70]:
        pref = 'LFI'
        hm = ['full', 'year-1-3', 'year-2-4']
        nside = '1024'
        ver = 'R2.01'
    else:
        pref = 'HFI'
        hm = ['full', 'halfmission-1', 'halfmission-2']
        nside = '2048'
        ver = 'R2.02'
        

    fn = []

    for k,val in enumerate(hm):
        fn.append(dir + '{:s}_SkyMap_{:03d}_{:s}_{:s}_{:s}.fits'.format(pref,f,nside,ver,val))
 
    return fn, nside

    
def dodg(fn_in, fn_out, nside=512, TQU=True, mask=None):

    udefval=-1.6375e+30
    if TQU:
        field = (0,1,2)
    else:
        field = 0

    hmap = np.array(hp.read_map(fn_in, field=field))

    if mask is not None:
        print('masking with {0}'.format(mask))
        mask = np.array(hp.read_map(mask, field=field))
        hmap[np.where(mask==udefval)] = udefval 

    hmap = hp.pixelfunc.ud_grade(hmap, nside, pess=False)
    hp.write_map(fn_out, hmap)


# Downgrade real map
fnr, nside = get_real_fnames(o.f) # real maps
fns        = get_mc_fnames(o.f, 0) # Choose rlz0 of mc_noise to create NaN mask
if o.f in [545, 857]:
    TQU = False
    field = (0)
else:
    TQU = True
    field = (0,1,2)


# Downgrade real maps
#for j,fn_in in enumerate(fnr):
#    print('downgrading {0}'.format(fn_in))
#    fn_out = fn_in.replace(nside, '512dg')
#    dodg(fn_in, fn_out, mask=fns[j], TQU=TQU)


rlz = o.rlz
fnn = get_mc_fnames(o.f, rlz)
for j,fn_in in enumerate(fnn):
    fn_out = fn_in.replace('map_mc','map_mc_512dg')
    # Skip if already exists
    if not os.path.isfile(fn_out):
        print('downgrading {0}'.format(fn_in))
        if o.f in [545, 857]:
            TQU = False
        else:
            TQU = True
        # Downgrade
        dodg(fn_in, fn_out, TQU=TQU)
    else:
        print('{0} already exists, skipping...'.format(fn_out))

