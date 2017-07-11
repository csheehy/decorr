import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import decorr


mapdir = '/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/'

# Load maps
map217a = np.array(hp.read_map(mapdir + 'real/HFI_SkyMap_217_2048_R2.02_halfmission-1.fits', field=(0,1,2)))
map217b = np.array(hp.read_map(mapdir + 'real/HFI_SkyMap_217_2048_R2.02_halfmission-2.fits', field=(0,1,2)))

map353a = np.array(hp.read_map(mapdir + 'real/HFI_SkyMap_353_2048_R2.02_halfmission-1.fits', field=(0,1,2)))
map353b = np.array(hp.read_map(mapdir + 'real/HFI_SkyMap_353_2048_R2.02_halfmission-2.fits', field=(0,1,2)))

# Downgrade
map217a = hp.ud_grade(map217a, 512)
map217b = hp.ud_grade(map217b, 512)
map353a = hp.ud_grade(map353a, 512)
map353b = hp.ud_grade(map353b, 512)

# Load LR63 mask, as in PIP L Fig 2
mask = np.array(hp.read_map(mapdir + 'masks/COM_Mask_Dust-diffuse-and-ps-PIP-L_0512_R2.00.fits', field=7))


# Get alms
alm217a = hp.map2alm(map217a*mask)
alm217b = hp.map2alm(map217b*mask)
alm353a = hp.map2alm(map353a*mask)
alm353b = hp.map2alm(map353b*mask)

# Get 217 x 353 for numerator in R_BB, mean of all four pairs
cl_1 = np.array(hp.alm2cl(alm217a, alm353a))
cl_2 = np.array(hp.alm2cl(alm217a, alm353b))
cl_3 = np.array(hp.alm2cl(alm217b, alm353a))
cl_4 = np.array(hp.alm2cl(alm217b, alm353b))
cl_cross = (cl_1 + cl_2 + cl_3 + cl_4)/4

# Get 217 x 217  and 353 x 353 c_ls for denominator
cl_217 = np.array(hp.alm2cl(alm217a, alm217b))
cl_353 = np.array(hp.alm2cl(alm353a, alm353b))

# Bin spectra like Planck
be = np.array([50., 160, 320, 500, 700])
bc = (be[0:-1] + be[1:])/2
specind = 2 # Use BB spectrum
l = np.arange(cl_217.shape[1])
cl_217_bin = []
cl_353_bin = []
cl_cross_bin = []
for k,val in enumerate(bc):
    ind = np.where( (l>=be[k]) & (l<be[k+1]) )
    cl_217_bin.append(cl_217[specind,ind].mean())
    cl_353_bin.append(cl_353[specind,ind].mean())
    cl_cross_bin.append(cl_cross[specind,ind].mean())

# Compute correlation coefficient
R = np.array(cl_cross_bin) / np.sqrt(np.array(cl_217_bin) * np.array(cl_353_bin))

# Get PIP L points
x = np.loadtxt('PlanckL_BB.csv', delimiter=',')


# Get PolSpice c_ls
maps = decorr.Maps()
maps.r = np.array([map217a, map353a, map217a, map353a, map217b, map353b])
s = decorr.Spec(maps)
cl_217_ps, cl_353_ps, cl_cross_ps, dum = s.getautocross_pspice(maps.r, mask)

# Get binned PolSpice
l = np.arange(auto1.shape[1])
cl_217_ps_bin = []
cl_353_ps_bin = []
cl_cross_ps_bin = []
for k,val in enumerate(bc):
    ind = np.where( (l>=be[k]) & (l<be[k+1]) )
    cl_217_ps_bin.append(cl_217_ps[specind,ind].mean())
    cl_353_ps_bin.append(cl_353_ps[specind,ind].mean())
    cl_cross_ps_bin.append(cl_cross_ps[specind,ind].mean())
R_ps = np.array(cl_cross_ps_bin) / np.sqrt(np.array(cl_217_ps_bin) * np.array(cl_353_ps_bin))

# Plot
plt.clf()
plt.plot(bc, R, 'o', label='R_BB, map2alm');
plt.plot(bc, R_ps, 'o', label='R_BB, PolSpice');
plt.plot(x[:,0], x[:,1], 'x', label='R_BB, PIP L')
plt.legend()
plt.show()
