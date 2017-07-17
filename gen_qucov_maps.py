import decorr
from optparse import OptionParser
import healpy as hp
import os

parser = OptionParser()
parser.add_option("-r", dest="rlz", type="int", default=0)
(o, args) = parser.parse_args()

ds = False

m = decorr.Maps()
maps = m.genqucov(ds=ds)

fn = m.get_map_filenames('mc_noise', rlz=o.rlz)
for k,val in enumerate(fn):
    fnout = val.replace('mc','qucov')

    # ds
    if ds:
        fnout = fnout.replace('hm','ds')

    dir = os.path.split(fnout)[0]
    if not os.path.exists(dir):
        os.mkdir(dir)

    if (ds) & ('full' in fnout):
        continue
    
    hp.write_map(fnout, maps[k])


