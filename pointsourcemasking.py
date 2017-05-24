    def loadgalmask(self, fsky):
        """Get galaxy map defined for fsky = 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.97, 0.99"""

        x={0.2:0, 0.4:1, 0.6:2, 0.7:3, 0.8:4, 0.9:5, 0.97:6, 0.99:7}
        hmap = hp.fitsfunc.read_map('maps/masks/HFI_Mask_GalPlane-apo5_2048_R2.00.fits',field=x[fsky])

        return hmap


    def prepmasks(self, type):
        """Get point source mask and galaxy mask and downgrade as
        necessary. Point source mask has added complication that when
        downgrading, the edge pixels of a hole convolve with the surrouding
        ones, making them 0-1. Set these to zero"""

        if type in ['ps','all']:
            print 'loading point source mask...'
            pm = self.loadpsmask()
            ind = np.where(pm==0)
            th,ph = hp.pix2ang(hp.npix2nside(pm.size), ind)
            pm0 = np.ones(hp.nside2npix(self.nside))
            ind = np.unique(hp.ang2pix(self.nside, th, ph))
            pm0[ind] = 0
            pm0 = hp.smoothing(pm0, fwhm=0.5*np.pi/180)
            self.pm = pm0


    def loadpsmask(self):
        """Get 217 and 353 point source masks and make a union of them"""

        hmap = hp.fitsfunc.read_map('maps/masks/HFI_Mask_PointSrc_2048_R2.00.fits',field=(2,3));
        hmap = hmap[0]*hmap[1]

        return hmap


    def apodizemask(self, mask, r_in, r_out):
        """Map space map apodization, fwhm in arcmin"""

        # Get list of pixels where mask = 0
        npix = mask.size
        nside = hp.npix2nside(npix)
        pix = np.arange(npix)
        ind = np.where(mask==0)
        pix0 = pix[ind]

        # Get lon/lat of zero pixels
        theta0,phi0 = hp.pix2ang(nside, pix0)

        # Get lon/lat of all pixels
        theta,phi = hp.pix2ang(nside, pix)

        # Build up mask by looping over zero pixels and replacing with a
        # smoothly apodized function
        mask_out = np.ones(mask.shape)
        pixsize_deg = 0.1145/(nside/512.0)
        npix = np.ceil(r_out/(pixsize_deg*60))

        for k,val in enumerate(pix0):
            lat1 = -theta0[k]+np.pi/2
            lon1 = phi0[k]

            # Get neighboring pixels
            doind = self.getneighbors(nside, val, npix)

            # Compute distance from zero pixel
            #lat2 = -theta[doind]+np.pi/2
            #lon2 = phi[doind]

            #r = self.distance(lat1, lon1, lat2, lon2)
            #r = r*180/np.pi*60 # rad->arcmin
            #w = -self.apfunc(r, r_in, r_out) + 1
            #mask_out[doind] = mask_out[doind] * w
            
            mask_out[doind] = 0

        return mask_out


    def apfunc(self, r, r_in, r_out):
        """An apodizing function that starts at one, transitions to a cosine at
        r=r_in, and is zero beyond r_out (both in arcmin). This is a Tukey window."""
        w = np.zeros(r.shape)
        w[r<=r_in]=1.0

        ind = (r>r_in) & (r<=r_out)
        rr = r[ind]
        w[ind] = 0.5*np.cos((rr-r_in)*np.pi/(r_out-r_in)) + 0.5
        
        return w


    def distance(self,lat1,lon1,lat2,lon2):
        """Compute great circle distance between two lat lon pairs on the unit
        sphere. Lat/lon input in radians"""
        cos = np.cos
        sin = np.sin
        atan2 = np.arctan2

        dlon = np.abs(lon2-lon1)
        dlat = np.abs(lat2-lat1)

        # Vincenty forumula (thanks wikipedia)
        a = np.sqrt( (cos(lat2)*sin(dlon))**2 +
                     (cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(dlon))**2 )
        b = sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(dlon)
        r = atan2(a,b)
        
        # Law of haversines, breaks down at certain points on sphere
        #r = 2*np.arcsin(np.sqrt(sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2))


        return r


    def getneighbors(self, nside, pix, npix):
        """Get neighboring pixels npix deep"""
        pixout = np.array([pix])
        for k in range(int(npix)):
            pix = hp.pixelfunc.get_all_neighbours(512,pixout)
            pixout = np.union1d(pixout, np.ravel(pix))
            pixout = pixout[pixout!=-1]
        return pixout

