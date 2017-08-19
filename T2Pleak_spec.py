import decorr
import healpy as hp
import numpy as np
import os
import cPickle as cP
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-l", dest="reg", type="str", default='LR53')

(o, args) = parser.parse_args()

m = decorr.Maps(reg=o.reg)
m.prepmaps('real')
m.prepmasks('all')

hmap217 = np.array(hp.read_map('maps/T2Pleakage_cds/T2QU_217_512.fits', field=(0,1,2)))
hmap353 = np.array(hp.read_map('maps/T2Pleakage_cds/T2QU_353_512.fits', field=(0,1,2)))

for k in [0,2,4]:
    m.r[k]   = hmap217
    m.r[k+1] = hmap353

s = decorr.Spec(m)
s.getspec('r', estimator='pspice')


# Now delete real for memory 
delattr(s,'maps')
delattr(s,'mask')

path = 'spec/T2Pleakage/'
if not os.path.exists(path):
    os.mkdir(path)

fn = '{0}.fits'.format(o.reg)

f = open('{0}/{1}'.format(path,fn), 'wb')
cP.dump(s, f)
f.close()
