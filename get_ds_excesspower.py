import decorr
import numpy as np
from matplotlib.pyplot import *
import healpy as hp
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-l", dest="reg", type="str", default='LR63')
(o, args) = parser.parse_args()

# Load half mission maps
hm1_217mc = np.array(hp.read_map('maps/mc_noise/217/ffp8_noise_217_hm1_map_mc_512dg_00000.fits',field=(0,1,2)))
hm2_217mc = np.array(hp.read_map('maps/mc_noise/217/ffp8_noise_217_hm2_map_mc_512dg_00000.fits',field=(0,1,2)))
hm1_353mc = np.array(hp.read_map('maps/mc_noise/353/ffp8_noise_353_hm1_map_mc_512dg_00000.fits',field=(0,1,2)))
hm2_353mc = np.array(hp.read_map('maps/mc_noise/353/ffp8_noise_353_hm2_map_mc_512dg_00000.fits',field=(0,1,2)))

ds1_217r = np.array(hp.read_map('maps/real/HFI_SkyMap_217-ds1_512dg_R2.02_full.fits',field=(0,1,2)))
ds2_217r = np.array(hp.read_map('maps/real/HFI_SkyMap_217-ds2_512dg_R2.02_full.fits',field=(0,1,2)))
ds1_353r = np.array(hp.read_map('maps/real/HFI_SkyMap_353-ds1_512dg_R2.02_full.fits',field=(0,1,2)))
ds2_353r = np.array(hp.read_map('maps/real/HFI_SkyMap_353-ds2_512dg_R2.02_full.fits',field=(0,1,2)))

# Construct differences
diff_217mc = hm1_217mc - hm2_217mc
diff_353mc = hm1_353mc - hm2_353mc

diff_217r = ds1_217r - ds2_217r
diff_353r = ds1_353r - ds2_353r

# Set LR region
val = o.reg

# Construct maps
map = decorr.Maps(reg=val)

# 217
x = np.array([diff_217mc, diff_217mc, diff_217r, diff_217r, diff_217r, diff_217r])
x[np.abs(x)==np.abs(decorr.udefval)] = np.nan
map.r = x
map.prepmasks('all')

s217 = decorr.Spec(map)
s217.getspec('r')


# 353
x = np.array([diff_353mc, diff_353mc, diff_353r, diff_217r, diff_353r, diff_353r])
x[np.abs(x)==np.abs(decorr.udefval)] = np.nan
map = decorr.Maps(reg=val)
map.r = x
map.prepmasks('all')

s353 = decorr.Spec(map)
s353.getspec('r')


# Bin
c217 = decorr.Calc(s217, bintype='lin',lmin=0,lmax=700,nbin=35)
c217.getR('r')
c353 = decorr.Calc(s353, bintype='lin',lmin=0,lmax=700,nbin=35)
c353.getR('r')

# Plot
l = c217.bc

clf();
subplot(2,2,1)
plot(l,c217.r[0,2,:],label='217x217 ds diff real')
plot(l,c217.r[3,2,:],label='217x217 hm diff MC')
legend(loc='upper left')
xlim(30,300)
ylim(0,5e-11)
ylabel('l(l+1)C_L^BB (K^2)')

subplot(2,2,2)
plot(l,c353.r[0,2,:],label='353x353 ds diff real')
plot(l,c353.r[3,2,:],label='353x353 hm diff MC')
xlim(30,300)
legend(loc='upper left')
ylim(0,1e-9)
ylabel('l(l+1)C_L^BB (K^2)')

subplot(2,2,3)
plot(l, c217.r[0,2,:] - c217.r[3,2,:], label='217 real - MC')
ylabel('diff')

subplot(2,2,4)
plot(l, c353.r[0,2,:] - c353.r[3,2,:], label='353 real - MC')
ylabel('diff')


savefig('ds_diff_{0}.png'.format(val))

# Now generate maps. Cl of excess noise in half mission map is 2x lower than excess noise in difference map. 
fac = l*(l+1)/(2*np.pi)
cl217 = (c217.r[0,:,:] - c217.r[3,:,:])/fac
cl353 = (c353.r[0,:,:] - c353.r[3,:,:])/fac

# Interpolate
lmax = 800
larr = np.arange(lmax)
cl217i = np.zeros((4,larr.size))
cl353i = np.zeros((4,larr.size))
for k in range(3):
    cl217i[k] = np.interp(larr, l, cl217[k])
    cl353i[k] = np.interp(larr, l, cl353[k])


nrlz = 100
for k in range(nrlz):
    noi217a = hp.synfast(cl217i, 512, new=True)
    noi217b = hp.synfast(cl217i, 512, new=True)
    noi353a = hp.synfast(cl353i, 512, new=True)
    noi353b = hp.synfast(cl353i, 512, new=True)

    hp.write_map('maps/excess_noise/{:s}_217_ds1_{:04d}.fits'.format(val,k), noi217a)
    hp.write_map('maps/excess_noise/{:s}_217_ds2_{:04d}.fits'.format(val,k), noi217a)
    hp.write_map('maps/excess_noise/{:s}_353_ds1_{:04d}.fits'.format(val,k), noi353a)
    hp.write_map('maps/excess_noise/{:s}_353_ds2_{:04d}.fits'.format(val,k), noi353a)


