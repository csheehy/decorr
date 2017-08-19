import healpy as hp
import numpy as np

freq = [217,353]

for f0 in freq:

    f = str(f0)

    tleakQ=hp.read_map("maps/T2Pleakage_cds/TLeakQ_"+f+"_ECP.fits")
    tleakU=hp.read_map("maps/T2Pleakage_cds/TLeakU_"+f+"_ECP.fits")

    psi,theta,phi= -3.9918090306328524E-004, 1.0504796200761286, 1.6822178082198811    
    almq=hp.map2alm(tleakQ)
    almu=hp.map2alm(tleakU)
    Nside=512
    hp.rotate_alm(almq,psi,theta,phi) ## rotate rotates in-place
    hp.rotate_alm(almu,psi,theta,phi) ## rotate rotates in-place
    tleakQg=hp.alm2map(almq,Nside)
    tleakUg=hp.alm2map(almu,Nside)

    trip=(np.zeros(len(tleakQg)),tleakQg,tleakUg)

    hp.write_map ("maps/T2Pleakage_cds/T2QU_"+f+"_512.fits",trip)
