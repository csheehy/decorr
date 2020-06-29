import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

beam={}
bdir='/gpfs/mnt/gpfs01/astro/workarea/csheehy/planckmaps/beams/'

#for f in [217, 353]:
for f in [100, 143]:

    if f==217:
        det = [5,6,7,8]
    elif f==353:
        det = [3,4,5,6]
    elif (f==100) | (f==143):
        det = [1,2,3,4]

    for d in det:
        for ab in ['a','b']:
            id=str(d)+str(ab)
            b=fits.getdata(bdir+"HFI_ScanBeam_"+str(f)+"-"+id+"_R2.00.fits")        
            beam[id]=b/b.sum() ## we want to normalize so that the sum is one

    if f==217:
        Tbeam=beam['5a'] # test beam
        Qbeam=0.5*((beam['5a']-beam['5b'])+(beam['6a']-beam['6b']))
        Ubeam=0.5*((beam['7a']-beam['7b'])+(beam['8a']-beam['8b']))
    elif f==353:
        Tbeam=beam['3a'] # test beam
        Qbeam=0.5*((beam['3a']-beam['3b'])+(beam['4a']-beam['4b']))
        Ubeam=0.5*((beam['5a']-beam['5b'])+(beam['6a']-beam['6b']))
    elif (f==100) | (f==143):
        Tbeam=beam['1a'] # test beam
        Qbeam=0.5*((beam['1a']-beam['1b'])+(beam['2a']-beam['2b']))
        Ubeam=0.5*((beam['3a']-beam['3b'])+(beam['4a']-beam['4b']))


    ## Now rotate 180 and average
    Qbeam=0.5*(Qbeam+np.rot90(Qbeam,k=2))
    Ubeam=0.5*(Ubeam+np.rot90(Ubeam,k=2))

    Navg=4
    def bavg (m,N):
        No=len(m)/N
        out=np.zeros((No,No))
        for i in range(No):
                for j in range(No):
                        out[i,j]=m[N*i:N*(i+1),N*j:N*(j+1)].sum()
        return out
    Qbeam_s=bavg(Qbeam[:,:],Navg)
    Ubeam_s=bavg(Ubeam[:,:],Navg)
    Tbeam_s=bavg(Tbeam[:,:],Navg)

    np.save(bdir+"TestBeam_"+str(f),Tbeam_s)
    np.save(bdir+"UBeam_"+str(f),Ubeam_s)
    np.save(bdir+"QBeam_"+str(f),Qbeam_s)

