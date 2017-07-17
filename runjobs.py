from decorr import Maps
from decorr import Spec
import string 
import os
import numpy as np
import cPickle as cP
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-r", dest="rlz", type="int", default=0)
parser.add_option("-p", dest="simprefix", type="str", default='gaussian_')
parser.add_option("-s", dest="sigtype", type="str", default='gaussian')
parser.add_option("-n", dest="noitype", type="str", default='qucov_noise')
parser.add_option("-l", dest="reg", type="str", default='LR53')
parser.add_option("-e", dest="est", type="str", default='pspice')

(o, args) = parser.parse_args()

comp = ['synch','therm']

m=Maps(nside=512, comp=comp, simprefix=o.simprefix, reg=o.reg);

# First get real and noise, needed for NaNmask
if 'ds' in o.noitype:
    m.prepmaps('realds')
else:
    m.prepmaps('real')
m.prepmaps(o.noitype, rlz=o.rlz)

# Now get masks
m.prepmasks('all')

# Now start getting spectra
s=Spec(m)
if o.rlz == 0:
    s.getspec('r', estimator=o.est)
delattr(m, 'r')

m.prepmaps(o.sigtype)
s.getspec('n', estimator=o.est)
s.getspec('s', estimator=o.est)
s.getspec(['s','n'], estimator=o.est)

# Now delete real for memory 
delattr(s,'maps')
delattr(s,'mask')

path = 'spec/{0}'.format(m.simprefix[0:-1])
if not os.path.exists(path):
    os.mkdir(path)

fn_prefix = string.join([val+'_' for k,val in enumerate(m.comp)], sep='')
fn_prefix = fn_prefix + str(m.reg)
if o.est == 'pspice':
    estsuffix = ''
else:
    estsuffix = o.est+'_'

fn = '{:s}_{:s}_{:s}{:04d}.pickle'.format(fn_prefix, o.noitype, estsuffix, o.rlz)

del m

f = open('{0}/{1}'.format(path,fn), 'wb')
cP.dump(s, f)
f.close()
