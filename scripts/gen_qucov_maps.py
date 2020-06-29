import decorr
from optparse import OptionParser
import healpy as hp
import os

parser = OptionParser()
parser.add_option("-r", dest="rlz", type="int", default=0)
parser.add_option("-t", dest="type", type="str", default="realhmds1")
(o, args) = parser.parse_args()

m = decorr.Maps()
maps = m.genqucov(type=o.type)

fn = m.get_map_filenames(o.type, rlz=o.rlz)
for k,val in enumerate(fn[2:]):

    fnout = val.replace('maps/real','maps/hmds/')
    fnout = fnout.replace('.fits','_noi{:05d}.fits'.format(o.rlz))

    dir = os.path.split(fnout)[0]
    if not os.path.exists(dir):
        os.mkdir(dir)

    hp.write_map(fnout, maps[k])


