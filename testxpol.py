import healpy as hp
import numpy as np
import decorr
import xpol_wrap as xpol

doload = False
mapdir = '/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/'

if doload:


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

    # Write maps
    hp.write_map('map217a.fits',map217a)
    hp.write_map('map217b.fits',map217b)
    hp.write_map('map353a.fits',map353a)
    hp.write_map('map353b.fits',map353b)
    hp.write_map('mask.fits',mask)
else:
    # Get spectra
    #fn217a = 'map217a.fits'
    #fn217b = 'map217b.fits'
    #fn353a = 'map353a.fits'
    #fn353b = 'map353b.fits'
    #fnmask = 'mask.fits'

    fn217a = mapdir + 'real/HFI_SkyMap_217_512dg_R2.02_halfmission-1.fits'
    fn217b = mapdir + 'real/HFI_SkyMap_217_512dg_R2.02_halfmission-2.fits'
    fn353a = mapdir + 'real/HFI_SkyMap_353_512dg_R2.02_halfmission-1.fits'
    fn353b = mapdir + 'real/HFI_SkyMap_353_512dg_R2.02_halfmission-2.fits'
    fnmask = 'mask.fits'

    map217a = hp.read_map(fn217a, field=(0,1,2))
    map217b = hp.read_map(fn217b, field=(0,1,2))
    map353a = hp.read_map(fn353a, field=(0,1,2))
    map353b = hp.read_map(fn353b, field=(0,1,2))
    mask    = hp.read_map(fnmask)

    mask = hp.read_map(fnmask)
    mask[map217b[2]<-1e20] = 0
    hp.write_map('mask.fits',mask)


# Get native binned Xpol
be = np.array([50., 160, 320, 500, 700])
bc = (be[0:-1] + be[1:])/2
l, auto217, clerr = xpol.getXcorr(512, fn217a, fn217b, fnmask, be)
l, auto353, clerr = xpol.getXcorr(512, fn353a, fn353b, fnmask, be)
l, crossa, clerr = xpol.getXcorr(512, fn217a, fn353a, fnmask, be)
l, crossb, clerr = xpol.getXcorr(512, fn217a, fn353b, fnmask, be)
l, crossc, clerr = xpol.getXcorr(512, fn217b, fn353a, fnmask, be)
l, crossd, clerr = xpol.getXcorr(512, fn217b, fn353b, fnmask, be)
cross = (crossa + crossb + crossc + crossd)/4

# Get raw Xpol, then bin
bee = np.arange(0,701)
l0, auto2170, clerr = xpol.getXcorr(512, fn217a, fn217b, fnmask, bee)
l0, auto3530, clerr = xpol.getXcorr(512, fn353a, fn353b, fnmask, bee)
l0, crossa0, clerr = xpol.getXcorr(512, fn217a, fn353a, fnmask, bee)
l0, crossb0, clerr = xpol.getXcorr(512, fn217a, fn353b, fnmask, bee)
l0, crossc0, clerr = xpol.getXcorr(512, fn217b, fn353a, fnmask, bee)
l0, crossd0, clerr = xpol.getXcorr(512, fn217b, fn353b, fnmask, bee)
cross0 = (crossa0 + crossb0 + crossc0 + crossd0)/4

cl_217_xp_bin = []
cl_353_xp_bin = []
cl_cross_xp_bin = []
specind = 2
for k,val in enumerate(bc):
    ind = np.where( (l0>=be[k]) & (l0<be[k+1]) )
    cl_217_xp_bin.append(auto2170[specind,ind].mean())
    cl_353_xp_bin.append(auto3530[specind,ind].mean())
    cl_cross_xp_bin.append(cross0[specind,ind].mean())

# Calculate R
R = cross / sqrt(auto217*auto353)
R0 = np.array(cl_cross_xp_bin) / sqrt(np.array(cl_217_xp_bin)*np.array(cl_353_xp_bin))

# Get PIP L points
x = np.loadtxt('PlanckL_BB.csv', delimiter=',')


# Get PolSpice c_ls
maps = decorr.Maps()
maps.r = np.array([map217a, map353a, map217a, map353a, map217b, map353b])
s = decorr.Spec(maps)
cl_217_ps, cl_353_ps, cl_cross_ps, dum = s.getautocross_pspice(maps.r, mask)

# Get binned PolSpice
ll = np.arange(cl_217_ps.shape[1])
cl_217_ps_bin = []
cl_353_ps_bin = []
cl_cross_ps_bin = []
specind = 2
for k,val in enumerate(bc):
    ind = np.where( (ll>=be[k]) & (ll<be[k+1]) )
    cl_217_ps_bin.append(cl_217_ps[specind,ind].mean())
    cl_353_ps_bin.append(cl_353_ps[specind,ind].mean())
    cl_cross_ps_bin.append(cl_cross_ps[specind,ind].mean())
R_ps = np.array(cl_cross_ps_bin) / np.sqrt(np.array(cl_217_ps_bin) * np.array(cl_353_ps_bin))

# Plot
clf()
plot(bc, R_ps, 'o', label='PolSpice');
plot(l, R[2], 'o', label='Xpol (natively binned)');
plot(bc, R0, '.', label='Xpol (binned after)');
plot(x[:,0], x[:,1], 'x', label='PIP L')
legend()
ylim(0.7,1.1)
grid('on')
xlim(50,700)

savefig('RBB_xpol_nanmask.png')
