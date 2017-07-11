import decorr
import numpy as np
import healpy as hp




def getRdbeta(dbeta, addlcdm):
    mod = decorr.Model()
    nside = 512

    nrlz = 2
    R = []
    for k in range(nrlz):
        dust353_map, dust353_alm = mod.gengalmap(f=353, fwhm=0)
        if addlcdm:
            lcdm_map, lcdm_alm = mod.genlcdmmap(fwhm=0)
        else:
            lcdm_map = 0

        fac = mod.freq_scale(217, nside=nside, dbeta=dbeta)
        dust217_map = dust353_map * fac

        alm353 = hp.map2alm(dust353_map + lcdm_map)
        alm217 = hp.map2alm(dust217_map + lcdm_map)

        cl217 = np.array(hp.alm2cl(alm217))
        cl353 = np.array(hp.alm2cl(alm353))
        clcross = np.array(hp.alm2cl(alm217, alm353))

        R.append((clcross / np.sqrt(cl217*cl353))[2])

    R = np.array(R)
    R = np.mean(R,0)

    # bin
    be = np.arange(0,701,10)
    bc = (be[0:-1] + be[1:])/2
    l = np.arange(R.size)
    Rbin = []
    for k,val in enumerate(bc):
        ind = (l>=be[k]) & (l<be[k+1]) 
        Rbin.append(R[ind].mean())
    Rbin = np.array(Rbin)

    return Rbin,bc,be

def getdbetascaling(db):
    f1 = 217.
    f2 = 353.
    fac = np.exp( -0.5 *db**2 * (np.log(f1/f2))**2 )
    return fac


getmodel = False


if getmodel:
    R0p5,bc,be = getRdbeta(0.5, False)
    R0p2,bc,be = getRdbeta(0.2, False)
    R0p1,bc,be = getRdbeta(0.1, False)
    R0p05,bc,be = getRdbeta(0.05, False)

    R0p5_lcdm,bc,be = getRdbeta(0.5, True)
    R0p2_lcdm,bc,be = getRdbeta(0.2, True)
    R0p1_lcdm,bc,be = getRdbeta(0.1, True)
    R0p05_lcdm,bc,be = getRdbeta(0.05, True)

ion()
clf()

mod = decorr.Model()
Rlcdm,dum = mod.getR(mod.fsky,be=be)

plot(bc, R0p5, 'b+', label='0.5')
plot(bc, R0p2, 'r+', label='0.2')
plot(bc, R0p1, 'g+', label='0.1')
plot(bc, R0p05, 'm+', label='0.05')

plot(bc, R0p5_lcdm, 'bx', label='0.5')
plot(bc, R0p2_lcdm, 'rx', label='0.2')
plot(bc, R0p1_lcdm, 'gx', label='0.1')
plot(bc, R0p05_lcdm, 'mx', label='0.05')

plot(bc, R0p5*Rlcdm, 'b.', label='0.5')
plot(bc, R0p2*Rlcdm, 'r.', label='0.2')
plot(bc, R0p1*Rlcdm, 'g.', label='0.1')
plot(bc, R0p05*Rlcdm, 'm.', label='0.05')

ylim(0.5,1)



