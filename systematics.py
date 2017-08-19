import decorr
import numpy as np
import healpy as hp
import ispice_wrap
import cPickle as cP

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-l", dest="reg", type="str", default='LR72')
(o, args) = parser.parse_args()

mapdir = 'maps/real/'

# Compute real hm1 - hm2 difference
hmdiff = {}
hm1 = {}
hm2 = {}
for f in [217, 353]:
    hm1[f] = np.array(hp.read_map(mapdir+'HFI_SkyMap_{:d}_512dg_R2.02_halfmission-1.fits'.format(f), field=(0,1,2)))
    hm2[f] = np.array(hp.read_map(mapdir+'HFI_SkyMap_{:d}_512dg_R2.02_halfmission-2.fits'.format(f), field=(0,1,2)))
    hm1[f][np.abs(hm1[f])>1e20] = 0
    hm2[f][np.abs(hm2[f])>1e20] = 0
    hmdiff[f] = (hm1[f]-hm2[f])/2
    hmdiff[f][~np.isfinite(hmdiff[f])] = 0
    hmdiff[f][np.abs(hmdiff[f])>1e20] = 0

dsdiff = {}
ds1 = {}
ds2 = {}
for f in [217, 353]:
    ds1[f] = np.array(hp.read_map(mapdir+'HFI_SkyMap_{:d}-ds1_512dg_R2.02_full.fits'.format(f), field=(0,1,2)))
    ds2[f] = np.array(hp.read_map(mapdir+'HFI_SkyMap_{:d}-ds2_512dg_R2.02_full.fits'.format(f), field=(0,1,2)))
    ds1[f][np.abs(ds1[f])>1e20] = 0
    ds2[f][np.abs(ds2[f])>1e20] = 0
    dsdiff[f] = (ds1[f]-ds2[f])/2
    dsdiff[f][~np.isfinite(dsdiff[f])] = 0
    dsdiff[f][np.abs(dsdiff[f])>1e20] = 0


# Compute FFP8 hm1 - hm2 difference
n = 20
hmsimdiff = {}
for f in [217,353]:
    mapdir = 'maps/mc_noise/{:d}/'.format(f)
    x = []
    for k in range(n):
        hm1sim = np.array(hp.read_map(mapdir+'ffp8_noise_{:d}_hm1_map_mc_512dg_{:05d}.fits'.format(f,k), field=(0,1,2)))
        hm2sim = np.array(hp.read_map(mapdir+'ffp8_noise_{:d}_hm2_map_mc_512dg_{:05d}.fits'.format(f,k), field=(0,1,2)))
        x.append( (hm1sim - hm2sim) / 2 )
    x = np.array(x)
    x[~np.isfinite(x)] = 0
    x[np.abs(x)>1e20] = 0
    hmsimdiff[f] = x

# Get mask
m = decorr.Maps()
mask = m.loadgalmask(o.reg)

# Get spectra
clhm_217 = np.array(ispice_wrap.ispice_wrap(hmdiff[217], hmdiff[217], mask))
clhm_353 = np.array(ispice_wrap.ispice_wrap(hmdiff[353], hmdiff[353], mask))
clffp8_217 = []
clffp8_353 = []
for k in range(hmsimdiff[217].shape[0]):
    clffp8_217.append(np.array(ispice_wrap.ispice_wrap(hmsimdiff[217][k], hmsimdiff[217][k], mask)))
    clffp8_353.append(np.array(ispice_wrap.ispice_wrap(hmsimdiff[353][k], hmsimdiff[353][k], mask)))
clffp8_217 = np.array(clffp8_217)
clffp8_353 = np.array(clffp8_353)

# Get frequency cross spectra
clhm_cross = np.array(ispice_wrap.ispice_wrap(hmdiff[217], hmdiff[353], mask))
clffp8_cross = []
for k in range(hmsimdiff[217].shape[0]):
    clffp8_cross.append(np.array(ispice_wrap.ispice_wrap(hmsimdiff[217][k], hmsimdiff[353][k], mask)))
clffp8_cross = np.array(clffp8_cross)

# Get cross spectra of hmdiff_f1 x map_f2
clhm_hmdiff217x353 = np.array(ispice_wrap.ispice_wrap(hmdiff[217], (hm1[353]+hm2[353])/2, mask))
clhm_217xhmdiff353 = np.array(ispice_wrap.ispice_wrap(hmdiff[353], (hm1[217]+hm2[217])/2, mask))

# Undo pixel window function and K^2 to uK^2
l = np.arange(clhm_217.shape[1])
pw = hp.pixwin(512)[0:(l.size)]

clhm_217 = clhm_217*pw**2*1e12
clffp8_217 = clffp8_217*pw**2*1e12
clhm_353 = clhm_353*pw**2*1e12
clffp8_353 = clffp8_353*pw**2*1e12
clhm_cross = clhm_cross*pw**2*1e12
clffp8_cross = clffp8_cross*pw**2*1e12


# Now compute decorrelation of hm and ds splits
mapdir = 'maps/real/'

m = decorr.Maps(reg=o.reg)
m.prepmaps('ds217')
m.prepmaps('ds353')
m.r = m.ds217 + m.ds353 # m.r is used when computing the NaN mask
m.prepmasks('all')

s = decorr.Spec(m)
s.getspec('ds217')
s.getspec('ds353')

delattr(s,'maps')
delattr(s,'mask')

data = {}
data['clhm_217'] = clhm_217
data['clffp8_217'] = clffp8_217
data['clhm_353'] = clhm_353
data['clffp8_353'] = clffp8_353
data['clhm_cross'] = clhm_cross
data['clffp8_cross'] = clffp8_cross
data['s'] = s


f = open('systematics_{:s}.pickle'.format(o.reg), 'wb')
cP.dump(data, f)
f.close()



