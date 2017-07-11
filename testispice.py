import decorr
import cPickle as cP
from optparse import OptionParser


parser = OptionParser()
parser.add_option("-r", dest="rlz", type="int", default=0)
parser.add_option("-l", dest="LR", type="str", default='LR63N')
(o, args) = parser.parse_args()

m = decorr.Maps(reg=o.LR)
m.prepmaps('gaussian')
m.prepmasks('all')

s = decorr.Spec(m)
s.getspec('s')

delattr(s,'maps')
delattr(s,'mask')

f = open('testispice/{:s}_{:04d}.pickle'.format(o.LR, o.rlz), 'wb')
cP.dump(s, f)
f.close()

