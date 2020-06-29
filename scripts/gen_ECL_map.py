import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

Nside=512
Npix=Nside**2*12
gmap=hp.read_map("/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/real/HFI_SkyMap_143_512dg_R2.02_full.fits")
gmap=hp.ud_grade(gmap,Nside)

psi,theta,phi=1.4593748453675195,1.0504796202735345,-3.1411934726869077
almg=hp.map2alm(gmap)
hp.rotate_alm(almg,psi,theta,phi) ## rotate rotates in-place
emap2=hp.alm2map(almg,Nside)

hp.write_map("/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/real/HFI_SkyMap_143_512dg_R2.02_full_ECL.fits",emap2)


