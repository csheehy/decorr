# Decorrelation of multifrequency CMB + dust maps as a function of ell
import numpy as np
import copy
from astropy.io import fits
import healpy as hp
import planck
from scipy.interpolate import interp1d
from matplotlib import pyplot as pp
from IPython.core.debugger import Tracer; debug_here=Tracer()

def genlcdmmap(fwhm=4.818,nside=2048):
    """Make a simulated LCDM TT+EE + diagonal lensing BB sky
    fwhm = beam FWHM in arcmin (default for Planck 353)"""

    # Get theory cl's
    cl,nm=readcambfits('camb_planck2013_r0_lensing.fits')

    # uK_cmb^2 -> K_cmb^2
    cl = cl*1e-12
    
    # TT, TE, EE, BB
    x=(cl[:,0],cl[:,3],cl[:,1],cl[:,2])

    # Make map and alms
    map,alm=hp.synfast(x, nside=nside, alm=True, pol=True, fwhm=(fwhm/60.0)*np.pi/180)

    return np.asarray(map),alm

def gengalmap(I353,fwhm=4.818,nside=2048):
    """Simulate a full sky 353 GHz dust map with Gaussian power"""

    # Dummy cl's
    cl,nm=readcambfits('camb_planck2013_r0_lensing.fits')
    
    # Get dust Cl's
    l=np.arange(2001)
    clE=getdustcl(l,353,I353,spec=1)*1e-12 #EE and uK^2->K^2
    clB=getdustcl(l,353,I353,spec=2)*1e-12 #EE and uK^2->K^2
    clT=clE/.05**2
    clTE=np.zeros(clT.shape)
    
    # TT, TE, EE, BB
    x=(clT,clTE,clE,clB)

    # Make map and alms
    map,alm=hp.synfast(x, nside=nside, alm=True, pol=True, fwhm=(fwhm/60.0)*np.pi/180)

    return np.asarray(map),alm
    
def readcambfits(fname):
    """Read in a CAMB generated fits file and return a numpy array of the table
    values. Returns numpy array of N_l x N_fields and a string array of field
    names. Returns C_l's in uK^2.""" 

    h=fits.open(fname)
    d=h[1].data
    nm=d.names

    nl=d[nm[0]].size
    nfields=np.size(nm)

    cl=np.zeros([nl,nfields])

    for k,val in enumerate(nm):
        cl[:,k]=d[val]

    # Convert to uK^2
    cl = cl*1e12
    
    return cl,nm

def getdustcl(l,f,I353,spec):
    """Kludge up a BB dust spectrum as a function of l at frequency f (in GHz) and
    <I_353> (in MJy sr^-1)."""

    # Dust D_l amplitude
    A_353 = I3532A(I353)

    # If EE, compensate for EE/BB=2
    if spec==1:
        A_353=A_353*1.923

    # Convert from D_l amplitude to C_l amplitude
    lfid = 80
    ffid = 353
    A_353 = 2*np.pi * A_353 / (lfid*(lfid+1))
    
    # From PIP XXX, spectrum is a power law in ell with -2.42 slope.
    cl = (l/80.0)**(-2.6)

    # Get dust scale factor
    sf = freq_scale(f,ffid)
    
    cl = A_353 * sf**2 *cl

    cl[l==0]=0;
    
    return cl

def freq_scale(f,ffid=353.0,beta=1.59,Tdust=19.6):
    """Conversion factor to scale dust map at frequency ffid (GHz) to f
    (GHz)."""

    # Make sure this works if f is an integer
    f = np.float(f)
    
    # Conversion factor for graybody
    fac = planck.planck(f*1e9,Tdust)/planck.planck(ffid*1e9,Tdust) * (f/ffid)**beta
    fac = fac.value
    
    # Conversion factor for thermodynamic temperature
    h = 6.626e-34
    kB = 1.381e-23
    Tcmb = 2.72548
    cf1 = f**4 * np.exp(h*f*1e9 / (kB*Tcmb)) / (np.exp(h*f*1e9 / (kB*Tcmb)) - 1)**2
    cf2 = ffid**4 * np.exp(h*ffid*1e9 / (kB*Tcmb)) / (np.exp(h*ffid*1e9 / (kB*Tcmb)) - 1)**2
    cf = cf1/cf2

    return fac/cf

def get_dust_params():

    fsky = np.array([0,.3,.4,.5,.6,.7,.8])
    I353_i = np.array([0,.068,.085,.106,.133,.167,.227])

def fsky2I353(fsky):
    """Interpolate I353 as a function of fsky from Planck XXX table. Outputs
    I353 in MJy sr^-1. Works in range 0<fsky<.8"""

    fsky_i = np.array([0,.3,.4,.5,.6,.7,.8])
    I353_i = np.array([0,.068,.085,.106,.133,.167,.227])
    f = interp1d(fsky_i,I353_i)
    I353 = f(fsky)

    return I353

def I3532fsky(I353):
    """Interpolate fsky as a function of I353 from Planck XXX table. Input I353
    in MJy sr^-1. Works in range 0<I353<.227"""

    fsky_i = np.array([0,.3,.4,.5,.6,.7,.8])
    I353_i = np.array([0,.068,.085,.106,.133,.167,.227])
    f = interp1d(I353_i,fsky_i)
    fsky = f(I353)

    return fsky

def I3532A(I353):
    """Return A, the dust D_l_BB amplitude at l=80 and f=353 GHz in uk_cmb^2, for
    input I_353 in MJy sr^-1. Currently am eyeballing A off a Planck XXX plot for
    fsky = 0.7, corresponding to I_353 = 0.167 MJy sr^-1. A varies as I_353^1.9."""

    # Dust D_l_BB amplitude at l=80 at 353 GHz in uK_CMB^2 for fsky=0.7
    lfid = 80.0
    ffid = 353.0
    Afid = 100.0
    Ifid = 0.167

    # From PIP XXX, fsky=0.7 corresponds to <I_353> = 0.167 MJy sr^-1. From
    # Aumont slides (and probably PIP XXX also), dust spectrum amplitude scales
    # as <I_dust>^1.9. Scale amplitude appropriately.
    A_353 = Afid * (I353/Ifid)**1.9

    return A_353

def A2I353(A):
    """Return I353 given input A (inverse of I3532A)"""

    # Dust D_l_BB amplitude at l=80 at 353 GHz in uK_CMB^2 for fsky=0.7
    lfid = 80.0
    ffid = 353.0
    Ifid = 0.167
    Afid = 100.0
    
    # From PIP XXX, fsky=0.7 corresponds to <I_353> = 0.167 MJy sr^-1. From
    # Aumont slides (and probably PIP XXX also), dust spectrum amplitude scales
    # as <I_dust>^1.9. Scale amplitude appropriately.
    I_353 = Ifid * (np.float(A)/Afid)**(1/1.9)

    return I_353

    
def getR(fsky,spec=2):
    """fsky = sky fraction
       spec = 0,1,2 for T,E,B. TT does not currently work because the dust power
              law needs to be defined. BICEP field corresponds to fsky=.135"""

    # Interpolate I_353 as a function of fsky
    I353 = fsky2I353(fsky)
    
    # Load lensing C_l in uK_cmb^2
    cl_l,nm = readcambfits('camb_planck2013_r0_lensing.fits');
    cl_l = cl_l[:,spec] # TT, EE, orBB
    l = np.arange(cl_l.size)

    # Get dust spectrum C_l in uK_cmb^2 at 353 and 217 GHz at fsky=0.5
    d_353 = getdustcl(l,353,I353,spec)
    d_217 = getdustcl(l,217,I353,spec)
    
    # Compute R = <353 x 217> / sqrt(<353x353><217x217>)
    #R = (np.sqrt(d_353)*np.sqrt(d_217) + cl_l) / np.sqrt((d_353 + cl_l + 2*d_353*cl_l)*(d_217+cl_l+2*d_217*cl_l))
    R = (np.sqrt(d_353)*np.sqrt(d_217) + cl_l) / np.sqrt((d_353 + cl_l)*(d_217+cl_l))

    #pp.semilogx(l,R);pp.xlim(10,1000);pp.ylim(.7,1.3);pp.grid('on');

    return R,l

def getpsmask():

    # Get 217 and 353 point source masks and make a union of them
    hmap = hp.fitsfunc.read_map('HFI_Mask_PointSrc_2048_R2.00.fits',field=(2,3));
    hmap = hmap[0]*hmap[1]

    ## Smooth to 10 arcmin FWHM
    #hmapsm = hp.sphtfunc.smoothing(hmap,fwhm=(10.0/60.0)*np.pi/180.)

    # Write to disk
    #hp.fitsfunc.write_map('mask_PS_217_353_union.fits',hmapsm)

    return hmap
    
def getgalmask(fsky):
    """fsky = 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.97, 0.99"""

    x={0.2:0, 0.4:1, 0.6:2, 0.7:3, 0.8:4, 0.9:5, 0.97:6, 0.99:7}
    hmap = hp.fitsfunc.read_map('HFI_Mask_GalPlane-apo5_2048_R2.00.fits',field=x[fsky])

    return hmap

def get_real_maps():
    """Get real Planck maps for cross correlation"""
        
    # Get 217 and 353 maps
    a,b,c = hp.fitsfunc.read_map('maps/HFI_SkyMap_217_2048_R2.02_full.fits',field=(0,1,2))
    hmap_217_full = [a,b,c]

    a,b,c = hp.fitsfunc.read_map('maps/HFI_SkyMap_353_2048_R2.02_full.fits',field=(0,1,2))
    hmap_353_full = [a,b,c]

    #a,b,c = hp.fitsfunc.read_map('maps/HFI_SkyMap_217_2048_R2.02_halfmission-1.fits',field=(0,1,2))
    a,b,c = hp.fitsfunc.read_map('maps/HFI_SkyMap_217-ds1_2048_R2.02_full.fits',field=(0,1,2))
    hmap_217_a = [a,b,c]

    #a,b,c = hp.fitsfunc.read_map('HFI_SkyMap_217_2048_R2.02_halfmission-2.fits',field=(0,1,2))
    a,b,c = hp.fitsfunc.read_map('maps/HFI_SkyMap_217-ds2_2048_R2.02_full.fits',field=(0,1,2))
    hmap_217_b = [a,b,c]

    #a,b,c = hp.fitsfunc.read_map('HFI_SkyMap_353_2048_R2.02_halfmission-1.fits',field=(0,1,2))
    a,b,c = hp.fitsfunc.read_map('maps/HFI_SkyMap_353-ds1_2048_R2.02_full.fits',field=(0,1,2))
    hmap_353_a = [a,b,c]

    #a,b,c = hp.fitsfunc.read_map('HFI_SkyMap_353_2048_R2.02_halfmission-2.fits',field=(0,1,2))
    a,b,c = hp.fitsfunc.read_map('maps/HFI_SkyMap_353-ds2_2048_R2.02_full.fits',field=(0,1,2))
    hmap_353_b = [a,b,c]

    return hmap_217_full,hmap_353_full,hmap_217_a,hmap_217_b,hmap_353_a,hmap_353_b
    
def get_psm_maps(do_lcdm=False):
    """Get Dunkly PSM maps"""

    #prefix = 'betaCOM_TdCOM_therm'
    prefix = 'betaCOM_TdCOM_sm1deg_cmb_freef_noise_spinn_synch_therm'
    
    if (not do_lcdm) & ('cmb' in prefix):
        print('error')
        stop
        
    if (do_lcdm) & ('cmb' not in prefix):
        ac,bc,cc = hp.read_map('../PySM/Ancillaries/CMB/taylens/lensed_cmb.fits',field=(0,1,2))
    else:
        ac=0; bc=0; cc=0;
        
    # Get 217 and 353 maps
    ad,bd,cd = hp.read_map('../PySM/Output_cannon/{0}_217p0_512.fits'.format(prefix),field=(0,1,2))
    hmap_217_full = [ad+ac,bd+bc,cd+cc]

    ad,bd,cd = hp.read_map('../PySM/Output_cannon/{0}_353p0_512.fits'.format(prefix),field=(0,1,2))
    hmap_353_full = [ad+ac,bd+bc,cd+cc]

    prefix = prefix.replace('sm1deg_','sm1deg_halfa_')
    ad,bd,cd = hp.read_map('../PySM/Output_cannon/{0}_217p0_512.fits'.format(prefix),field=(0,1,2))
    hmap_217_a = [ad+ac,bd+bc,cd+cc]

    ad,bd,cd = hp.read_map('../PySM/Output_cannon/{0}_353p0_512.fits'.format(prefix),field=(0,1,2))
    hmap_353_a = [ad+ac,bd+bc,cd+cc]

    prefix = prefix.replace('halfa','halfb')
    ad,bd,cd = hp.read_map('../PySM/Output_cannon/{0}_217p0_512.fits'.format(prefix),field=(0,1,2))
    hmap_217_b = [ad+ac,bd+bc,cd+cc]

    ad,bd,cd = hp.read_map('../PySM/Output_cannon/{0}_353p0_512.fits'.format(prefix),field=(0,1,2))
    hmap_353_b = [ad+ac,bd+bc,cd+cc]

    return hmap_217_full,hmap_353_full,hmap_217_a,hmap_217_b,hmap_353_a,hmap_353_b
    

def get_sim_maps(Adust,do_lcdm=False):
    """Get simulated dust maps, e.g. just dust map at 353 GHz scaled, and an
    LCDM map.  do_lcdm = add a simulated LCDM map to the output dust maps

    """

    # Nside of sim
    Nside = 256    
    fwhm = 10.0

    # Optionally explore wierd dust distrubutions
    pix=np.arange(hp.nside2npix(Nside));
    theta,phi=hp.pix2ang(Nside,pix)
    b=-theta*180/np.pi+90

    # Dust as function of latitude
    Aarr=np.array([5,10,20,50,100]);
    barr=np.array([80,70,50,30,10])

    for k,val in enumerate(Aarr):

        I353=A2I353(val)
        map,alm=gengalmap(I353,fwhm=fwhm,nside=Nside)

        if k==0:
            hmap_353_full=copy.deepcopy(map)
        else:
            for j in range(3):
                ind = np.abs(b)<barr[k]
                hmap_353_full[j][ind]=map[j][ind];

    # Normalize map
    alm=hp.map2alm(hmap_353_full)
    cl=hp.alm2cl(alm)
    fac = Adust / (1e12*cl[2][80]*80*(81) / (2*np.pi))

    for k in range(3):
        hmap_353_full[k]=hmap_353_full[k]*np.sqrt(fac)
    
    # Copy to other maps
    hmap_217_full = copy.deepcopy(hmap_353_full)

    # 353 -> 217 scale factor
    fac = freq_scale(217.0)
    for k in range (0,3):
        hmap_217_full[k]=hmap_217_full[k]*fac

    # LCDM map
    hmap_lcdm,alm = genlcdmmap(fwhm=fwhm,nside=Nside)

    if do_lcdm:
        hmap_353_full = addmaps(hmap_353_full,hmap_lcdm)
        hmap_217_full = addmaps(hmap_217_full,hmap_lcdm)
        
    hmap_353_a = copy.deepcopy(hmap_353_full)
    hmap_353_b = copy.deepcopy(hmap_353_full)
    hmap_217_a = copy.deepcopy(hmap_217_full)
    hmap_217_b = copy.deepcopy(hmap_217_full)
    
    # Add in systematics
    #qa,ua=hp.fitsfunc.read_map('HFI_CorrMap_353-leakage-global_2048_R2.00_year-1.fits',field=(0,1));
    #qb,ub=hp.fitsfunc.read_map('HFI_CorrMap_353-leakage-global_2048_R2.00_year-2.fits',field=(0,1));
    #q,u=hp.fitsfunc.read_map('HFI_CorrMap_353-leakage-global_2048_R2.00_full.fits',field=(0,1));
    #qa[qa<1e-20]=0
    #qb[qb<1e-20]=0
    #ua[ua<1e-20]=0
    #ub[ub<1e-20]=0
    #q[q<1e-20]=0
    #u[u<1e-20]=0
    
    #hmap_353_full[1]=hmap_353_full[1]+q
    #hmap_353_full[2]=hmap_353_full[2]+u
    #hmap_353_a[1]=hmap_353_a[1]+qa
    #hmap_353_a[2]=hmap_353_a[2]+ua
    #hmap_353_b[1]=hmap_353_b[1]+qb
    #hmap_353_b[2]=hmap_353_b[2]+ub
    #fac = 1
    #hmap_217_full[1]=hmap_217_full[1]+q*fac
    #hmap_217_full[2]=hmap_217_full[2]+u*fac
    #hmap_217_a[1]=hmap_217_a[1]+qa*fac
    #hmap_217_a[2]=hmap_217_a[2]+ua*fac
    #hmap_217_b[1]=hmap_217_b[1]+qb*fac
    #hmap_217_b[2]=hmap_217_b[2]+ub*fac
    
    
    return hmap_217_full,hmap_353_full,hmap_217_a,hmap_217_b,hmap_353_a,hmap_353_b,hmap_lcdm

def addmaps(map1,map2):
    """Add two healpix maps together"""
    
    for k,val in enumerate(map1):
        map1[k] = map1[k]+map2[k]

    return map1


def calccorr(fsky):

    # Get point source mask and galaxy mask
    fsky =  0.6
    pm = getpsmask()
    gm = getgalmask(fsky)

    # Get real maps
    #hmap_217_full,hmap_353_full,hmap_217_a,hmap_217_b,hmap_353_a,hmap_353_b = get_real_maps()
    #m = pm*gm
    #dodg = True

    # Get "simulated" dust maps (e.g. just scaled versions of 353) and LCDM +
    # lensing BB
    #do_lcdm = True # Add LCDM power on top of the dust
    #Adust=50.0
    #hmap_217_full,hmap_353_full,hmap_217_a,hmap_217_b,hmap_353_a,hmap_353_b,hmap_lcdm = get_sim_maps(Adust,do_lcdm=do_lcdm)
    #m = 1.0;
    #dodg = False

    # Get PSM maps
    do_lcdm = True
    hmap_217_full,hmap_353_full,hmap_217_a,hmap_217_b,hmap_353_a,hmap_353_b = get_psm_maps(do_lcdm=do_lcdm)
    m = hp.pixelfunc.ud_grade(gm,512)
    dodg = False

    # Multiply by mask
    for k,val in enumerate(hmap_217_full):
        hmap_217_full[k] = hmap_217_full[k]*m
        hmap_353_full[k] = hmap_353_full[k]*m
        hmap_217_a[k] =  hmap_217_a[k]*m
        hmap_217_b[k] =  hmap_217_b[k]*m
        hmap_353_a[k] =  hmap_353_a[k]*m
        hmap_353_b[k] =  hmap_353_b[k]*m     
    
    # Optionally downgrade maps
    if dodg:
        Nside=512
        hmap_217_full = hp.pixelfunc.ud_grade(hmap_217_full,Nside)
        hmap_353_full = hp.pixelfunc.ud_grade(hmap_353_full,Nside)
        hmap_217_a = hp.pixelfunc.ud_grade(hmap_217_a,Nside)
        hmap_217_b = hp.pixelfunc.ud_grade(hmap_217_b,Nside)
        hmap_353_a = hp.pixelfunc.ud_grade(hmap_353_a,Nside)
        hmap_353_b = hp.pixelfunc.ud_grade(hmap_353_b,Nside)
    
    # Get alms
    alm_217_full = hp.sphtfunc.map2alm(hmap_217_full,pol=True,lmax=1024)
    alm_353_full = hp.sphtfunc.map2alm(hmap_353_full,pol=True,lmax=1024)
    alm_217_a = hp.sphtfunc.map2alm(hmap_217_a,pol=True,lmax=1024)
    alm_217_b = hp.sphtfunc.map2alm(hmap_217_b,pol=True,lmax=1024)
    alm_353_a = hp.sphtfunc.map2alm(hmap_353_a,pol=True,lmax=1024)
    alm_353_b = hp.sphtfunc.map2alm(hmap_353_b,pol=True,lmax=1024)

    # 217x353
    cross = [[],[],[]]
    auto217 = [[],[],[]]
    auto353 = [[],[],[]]

    
    for k,val in enumerate(cross):
        cross[k]=hp.alm2cl(alm_217_full[k],alm_353_full[k])
        auto217[k]=hp.alm2cl(alm_217_a[k],alm_217_b[k])
        auto353[k]=hp.alm2cl(alm_353_a[k],alm_353_b[k])

    for dum,spec in enumerate(np.array([1,2])):

        # Bin logarithmically
        if spec==1:
            nbin=100
        elif spec==2:
            nbin=500
            #nbin=7
        
        le=np.logspace(1,np.log10(950),nbin+1)
        #le=np.logspace(1,np.log10(800),nbin+1)
        bc=np.zeros(le.size-1)
        for k in range(0,le.size-1):
            bc[k]=(le[k]+le[k+1])/2

        x=np.zeros(bc.size)
        crossbin = [x,copy.deepcopy(x),copy.deepcopy(x)]
        auto217bin = copy.deepcopy(crossbin)
        auto353bin = copy.deepcopy(crossbin)

        l=np.arange(0,1025)

        for k in range(0,3):
            for j in range(0,bc.size):
                crossbin[k][j]=np.mean(cross[k][np.logical_and(l>le[j],l<le[j+1])])
                auto217bin[k][j]=np.mean(auto217[k][np.logical_and(l>le[j],l<le[j+1])])
                auto353bin[k][j]=np.mean(auto353[k][np.logical_and(l>le[j],l<le[j+1])])

    
        R=crossbin[spec]/np.sqrt(auto353bin[spec]*auto217bin[spec])
        if spec==1:
            REpl = copy.deepcopy(R)
            bcE = copy.deepcopy(bc)
        elif spec==2:
            RBpl = copy.deepcopy(R)
            bcB = copy.deepcopy(bc)
 
        #pp.semilogx(bc,R,'o');pp.ylim(0.7,1.3);pp.xlim(10,1000);pp.grid('on');

        # Get decorr B-mode model
        #Rmod,lmod = getR(fsky)

    return REpl, bcE, RBpl, bcB



    
