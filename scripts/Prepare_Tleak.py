#!/usr/bin/env python

#
# wq sub -r "N:204; threads:1;hostfile:auto" -c "a_mpirun -hostfile %hostfile% ./Prepare_Tleak.py U 217"
#

import healpy as hp
import numpy as np
import healconv as hc
import sys
from mpi4py import MPI 


QU = sys.argv[1]
f = sys.argv[2]
bdir = 'maps/beams/'

if QU=='U':
    Xbeam=np.load("maps/beams/UBeam_"+f+".npy")
elif QU=='Q':
    Xbeam=np.load("maps/beams/QBeam_"+f+".npy")
elif QU=='T':
    Xbeam=np.load("maps/beams/TestBeam_"+f+".npy")
else:
    print "Bad option"
    sys.exit(1)
DX=3.87850944888e-05

Tmape=hp.read_map('maps/real/HFI_SkyMap_'+f+'_512dg_R2.02_full_ECL.fits')
Nside=512

Leak=hc.healconvolve(Tmape, Xbeam, DX, 32, 1.4, MPI=MPI)

if Leak is not None:
    hp.write_map("maps/T2Pleakage_cds/TLeak"+QU+"_"+f+"_ECP.fits",Leak)


