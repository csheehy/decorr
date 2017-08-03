#!/usr/bin/env python

#
# wq sub -r "N:204; threads:4;hostfile:auto" -c "a_mpirun -hostfile %hostfile% ./Prepare_Tleak.py U"
#

import healpy as hp
import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
import healconv as hc
import sys
from mpi4py import MPI 


QU=sys.argv[1]
if QU=='U':
    Xbeam=np.load("/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/beams/UBeam_217.npy")
elif QU=='Q':
    Xbeam=np.load("/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/beams/QBeam_217.npy")
elif QU=='T':
    Xbeam=np.load("/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/beams/TestBeam5A_217.npy")
else:
    print "Bad option"
    sys.exit(1)
DX=3.87850944888e-05

Tmape=hp.read_map('HFI_SkyMap_217_512dg_R2.02_full_ECL.fits')
Nside=512

Leak=hc.healconvolve(Tmape,Xbeam,DX,32,1.4,MPI=MPI)

if Leak is not None:
    hp.write_map("TLeak"+QU+".fits",Leak)


