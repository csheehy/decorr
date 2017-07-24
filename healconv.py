#
# module to convolve a healpix map with a small 2D beam
#
import numpy as np
import healpy as hp
from numpy.fft import rfft2, irfft2
import matplotlib.pyplot as plt
from numpy.fft import rfft2,irfft2

def healconvolve (hmap, beam, dang, Nside_gn=4, ext_fact=1.5,debug=0):
    """Input:
       hmap -- input healpy map, an np.array
       beam -- a 2D image of the beam
       dang -- distance in radians between two pixels in beam
       Nside_gn -- we'll do gnomonic projection on this smaller Nside grid
       ext_factor -- how big do we make each 'patch'
       debug -- (1=don't actuall convolve, just deproject back), (2 - make colored patches)
    """

    ## first determine the input map Nside
    Nside=int(np.sqrt(len(hmap)/12))
    Npix=Nside**2*12
    Npix_gn=Nside_gn**2*12
    assert (len(hmap)==Npix)
    outmap=np.zeros(Npix)
    outw=np.zeros(Npix)
    ## Let's try to get the size of projection patch.
    side_rad=np.sqrt(4*np.pi/Npix_gn)*ext_fact   ## 1.2 is the safety factor, to be determined later
    Npr=int(side_rad/dang)
    ## fft beam, together with padding
    beam_=np.zeros((3*Npr,3*Npr))
    Nb=len(beam)/2
    beam_[-Nb:,-Nb:]=beam[0:Nb,0:Nb]
    beam_[-Nb:,:Nb]=beam[0:Nb,Nb:]
    beam_[:Nb,-Nb:]=beam[Nb:,0:Nb]
    beam_[:Nb,:Nb]=beam[Nb:,Nb:]
    #plt.figure()
    #plt.imshow(beam_)
    #return None
    beamfft=rfft2(beam_)
    vec2pix=lambda x,y,z:hp.vec2pix(Nside,x,y,z)
    ilist=np.outer(range(Npr),np.ones(Npr,int))
    jlist=ilist.T
    ilist=ilist.flatten()
    jlist=jlist.flatten()
    thetal,phil=hp.pix2ang(Nside_gn,range(Npix_gn))
    print Npr, len(ilist)
    cc=0
    for theta_p, phi_p in zip(thetal,phil):
        rot=(phi_p*180/np.pi, 90-theta_p*180/np.pi,0.)
        # forward and backward projectors
        projf=hp.projector.GnomonicProj(xsize = 3*Npr, ysize = 3*Npr, rot = rot, reso = dang*180*60/np.pi)
        projb=hp.projector.GnomonicProj(xsize = Npr, ysize = Npr, rot = rot, reso = dang*180*60/np.pi)
        proj2d=projf.projmap(hmap,vec2pix) ## 2D projection
        ## now we do the convolution
        if debug==1:
            proj2d=proj2d[Npr:2*Npr,Npr:2*Npr]
        elif debug==2:
            proj2d=np.ones((Npr,Npr))*cc
            cc+=1
        else:
            ## do the actual convolution
            ## first pad with zeros
            if (debug==3):
                plt.figure()
                plt.imshow(proj2d)
                plt.colorbar()
            ## convolve
            big=irfft2(rfft2(proj2d)*beamfft)
            proj2d=big[Npr:2*Npr,Npr:2*Npr]
            if (debug==3):
                plt.figure()
                plt.imshow(big)
                plt.colorbar()

        ## and copy back
        xl,yl=projb.ij2xy(ilist, jlist)
        theta,phi=projb.xy2ang(xl,yl)
        pixlist=proj2d[ilist,jlist]
        ndx=hp.ang2pix(Nside,theta,phi)
        outmap[ndx]+=pixlist
        outw[ndx]+=1
    outmap/=outw
    return outmap

        
    

