import decorr
from optparse import OptionParser
import healpy as hp
import os

parser = OptionParser()
parser.add_option("-r", dest="rlz", type="int", default=0)
(o, args) = parser.parse_args()

m = decorr.Maps()
m.prepmaps('qucov_noise')
fn = m.get_map_filenames('mc_noise', rlz=o.rlz)
for k,val in enumerate(fn):
    fnout = val.replace('mc','qucov')

    dir = os.path.split(fnout)[0]
    if not os.path.exists(dir):
        os.mkdir(dir)

    hp.write_map(fnout, m.n[k])


