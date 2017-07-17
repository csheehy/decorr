import healpy as hp
import decorr
import numpy as np

doload = True
if doload:
    m = decorr.Maps()
    m.prepmaps('real')

# Real 217
Tg = m.r[0,0,:]
pix = np.arange(pix.size)

# Coordinates of map
theta,phi = hp.pix2ang(512,pix)

# Healpix lat/lon
lat = 90 - theta*180/np.pi
lon = phi*180/np.pi

# If the map is an ecliptic rotation, what are the galactic coords?
l, b = hp.rotator.euler(lon, lat, 5)

# Pixel list to rotate galactic map to ecliptic map
pix_g2e = hp.ang2pix(512, (90-b)*np.pi/180, l*np.pi/180)

# If the map is a galactic rotation, what are the ecliptic coords?
elon, elat = hp.rotator.euler(lon, lat, 6)

# Pixel list to rotate galactic map to ecliptic map
pix_e2g = hp.ang2pix(512, (90-elat)*np.pi/180, elon*np.pi/180)


# Nearest neighbor interpolation to ecliptic rotated map
Te = Tg[pix_g2e]

# Get derivative map
alm = hp.map2alm(Te)
dum, dth, dph = hp.alm2map_der1(alm, 512)
dth_alm = hp.map2alm(dth)
dph_alm = hp.map2alm(dph)
dth, dth2, dthdph = hp.alm2map_der1(dth_alm, 512)
dum, dum, dph2 = hp.alm2map_der1(dph_alm, 512)

#############
# Construct leakage templates
lkt = np.zeros( (6, Te.size) )

lkt[0] = Te # Gain
lkt[1] = dph # diff point x
lkt[2] = dth # diff point y
lkt[3] = (dph2 + dth2) # diff bw
lkt[4] = (dph2 - dth2) # plus ellip
lkt[5] = 2*dthdph # cross ellip

#############
# Now rotate leakage template back to galactic coordinates
LKT = lkt[:, pix_e2g]





