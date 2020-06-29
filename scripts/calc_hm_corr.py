import decorr
import numpy as np
import cPickle as cP
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-r", dest="rlz", type="int", default=0)
parser.add_option("-f", dest="f", type="int", default=217)
(o, args) = parser.parse_args()

m=decorr.Maps(reg='LR72')

m.prepmaps('gaussian')
m.prepmaps('mcplusexcess_noise',rlz=o.rlz)
m.prepmaps('real')

# Rearrange to take hm1xhm2, hm1xhm1, hm2xhm2
# Order is [217f, 353f, 217hm1, 353hm1, 217hm2, 353hm2]
# Want [217hm1, 217hm2, 217hm1, 217hm2, 217hm1, 217hm2]

if o.f == 217:
    ind = [2, 4, 2, 4, 2, 4] # 217
if o.f == 353:
    ind = [3, 5, 3, 5, 3, 5] # 353

m.r = m.r[ind]
m.s = m.s[ind]
m.n = m.n[ind]

m.prepmasks('all')


s=decorr.Spec(m)

if o.rlz==0:
    s.getspec('r')
s.getspec(['s','n'])

delattr(s,'maps')
delattr(s,'mask')

# Bin
#s.sn = np.reshape(s.sn,[1,4,3,701])
#c = decorr.Calc(s, bintype='lin', lmin=0, lmax=700, nbin=70, full=True)
#c.getR('r')
#c.getR('sn')

# Save
path = 'spec/hmcorr'
fn = 'corr_hm1xhm2_{:03d}_{:04d}.pickle'.format(o.f,o.rlz)

f = open('{0}/{1}'.format(path,fn), 'wb')
cP.dump(s, f)
f.close()


