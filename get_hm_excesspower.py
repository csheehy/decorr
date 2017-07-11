import decorr
import numpy as np
from matplotlib.pyplot import *
import healpy as hp

# Load real maps
hm1_217r = np.array(hp.read_map('maps/real/HFI_SkyMap_217_512dg_R2.02_halfmission-1.fits',field=(0,1,2)))
hm2_217r = np.array(hp.read_map('maps/real/HFI_SkyMap_217_512dg_R2.02_halfmission-2.fits',field=(0,1,2)))
hm1_353r = np.array(hp.read_map('maps/real/HFI_SkyMap_353_512dg_R2.02_halfmission-1.fits',field=(0,1,2)))
hm2_353r = np.array(hp.read_map('maps/real/HFI_SkyMap_353_512dg_R2.02_halfmission-2.fits',field=(0,1,2)))

# Load mc maps
nrlz = 10
hm1_217mc = []
hm2_217mc = []
hm1_353mc = []
hm2_353mc = []
for k in range(nrlz):
    hm1_217mc.append(hp.read_map('maps/mc_noise/217/ffp8_noise_217_hm1_map_mc_512dg_{:05d}.fits'.format(k),field=(0,1,2)))
    hm2_217mc.append(hp.read_map('maps/mc_noise/217/ffp8_noise_217_hm2_map_mc_512dg_{:05d}.fits'.format(k),field=(0,1,2)))
    hm1_353mc.append(hp.read_map('maps/mc_noise/353/ffp8_noise_353_hm1_map_mc_512dg_{:05d}.fits'.format(k),field=(0,1,2)))
    hm2_353mc.append(hp.read_map('maps/mc_noise/353/ffp8_noise_353_hm2_map_mc_512dg_{:05d}.fits'.format(k),field=(0,1,2)))
hm1_217mc = np.array(hm1_217mc)
hm2_217mc = np.array(hm2_217mc)
hm1_353mc = np.array(hm1_353mc)
hm2_353mc = np.array(hm2_353mc)

# Construct differences
diff_217mc = hm1_217mc - hm2_217mc
diff_353mc = hm1_353mc - hm2_353mc

diff_217r = hm1_217r - hm2_217r
diff_353r = hm1_353r - hm2_353r

val = 'LR72'
map = decorr.Maps(reg=val)
gm = map.loadgalmask(val)

# Get spectra
normfac = 1/np.mean(gm**2)
diff_217r[np.abs(diff_217r) > 1e20] = 0
diff_353r[np.abs(diff_353r) > 1e20] = 0
cl217r = np.array(hp.alm2cl(hp.map2alm(diff_217r*gm)))/normfac
cl353r = np.array(hp.alm2cl(hp.map2alm(diff_353r*gm)))/normfac


cl217mc = []
cl353mc = []
diff_217mc[np.abs(diff_217mc) > 1e20] = 0
diff_353mc[np.abs(diff_353mc) > 1e20] = 0
for k in range(nrlz):
    print k
    cl217mc.append(np.array(hp.alm2cl(hp.map2alm(diff_217mc[k]*gm)))/normfac)
    cl353mc.append(np.array(hp.alm2cl(hp.map2alm(diff_353mc[k]*gm)))/normfac)
cl217mc = np.array(cl217mc)
cl353mc = np.array(cl353mc)

# Average mc 
cl217mcmean = cl217mc.mean(0)
cl353mcmean = cl353mc.mean(0)



# Bin Cl (not l^2 cl)
be = np.arange(0,800,20)
bc = (be[0:-1] + be[1:])/2.
l = np.arange(cl217r.shape[1])
nbin = be.size-1

cl217rbin = np.zeros((6,nbin))
cl353rbin = np.zeros((6,nbin))
cl217mcbin = np.zeros((6,nbin))
cl353mcbin = np.zeros((6,nbin))
for k in range(be.size-1):
    ind = (l>=be[k]) & (l<be[k+1])
    cl217rbin[:,k] = np.mean(cl217r[:,ind],1)
    cl353rbin[:,k] = np.mean(cl353r[:,ind],1)
    cl217mcbin[:,k] = np.mean(cl217mcmean[:,ind],1)
    cl353mcbin[:,k] = np.mean(cl353mcmean[:,ind],1)


excess217 = cl217rbin - cl217mcbin
excess353 = cl353rbin - cl353mcbin


# Plot
l = bc
fac = l*(l+1)/(2*np.pi)

clf();
subplot(2,2,1)
plot(l, cl217rbin[2]*fac, label='217x217 hm diff real')
plot(l, cl217mcbin[2]*fac, label='217x217 hm diff MC')
legend(loc='upper left')
xlim(30,700)
ylabel('l(l+1)C_L^BB (K^2)')

subplot(2,2,2)
plot(l,cl353rbin[2]*fac,label='353x353 hm diff real')
plot(l,cl353mcbin[2]*fac,label='353x353 hm diff MC')
xlim(30,700)
legend(loc='upper left')
ylabel('l(l+1)C_L^BB (K^2)')

subplot(2,2,3)
plot(l, excess217[2]*fac, label='217 real - MC')
ylabel('diff')

subplot(2,2,4)
plot(l, excess353[2]*fac, label='353 real - MC')
ylabel('diff')


#figure(2)
#plot(l, c353.r[1,2,:],label='(217 hm diff) x (353 hm diff)')
#legend()

savefig('hm_diff.png'.format(val))

# Now generate maps. Cl of excess noise in half mission map is 2x lower than excess noise in difference map. 
fac = l*(l+1)/(2*np.pi)
cl217 = excess217/2
cl353 = excess217/2

# Interpolate
lmax = 750
larr = np.arange(lmax)
cl217i = np.zeros((4,larr.size))
cl353i = np.zeros((4,larr.size))
for k in range(3):
    cl217i[k] = np.interp(larr, l, cl217[k])
    cl353i[k] = np.interp(larr, l, cl353[k])

nrlz = 1000
for k in range(nrlz):
    
    print k

    noi217a = hp.synfast(cl217i, 512, new=True)
    noi217b = hp.synfast(cl217i, 512, new=True)
    noi353a = hp.synfast(cl353i, 512, new=True)
    noi353b = hp.synfast(cl353i, 512, new=True)

    hp.write_map('maps/excess_noise/hm1_217_{:04d}.fits'.format(k), noi217a)
    hp.write_map('maps/excess_noise/hm2_217_{:04d}.fits'.format(k), noi217b)
    hp.write_map('maps/excess_noise/hm1_353_{:04d}.fits'.format(k), noi353a)
    hp.write_map('maps/excess_noise/hm2_353_{:04d}.fits'.format(k), noi353b)


