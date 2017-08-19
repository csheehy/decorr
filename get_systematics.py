import decorr
import cPickle as cP
import numpy as np


def get_excess(r, sim):
    # Input l
    lin = np.arange(r.shape[1])

    # Excess
    cl_excess = r - sim.mean(0)
    
    # Now bin
    be = np.arange(0,701,10)
    bc = (be[0:-1] + be[1:])/2.0
    cl_bin = np.zeros( (cl_excess.shape[0], bc.size) )
    for k,val in enumerate(bc):
        ind = (lin>=be[k]) & (lin<be[k+1])
        cl_bin[:,k] = cl_excess[:,ind].mean(1)

    # Now interpolate back to input l
    cl_out = np.zeros(cl_excess.shape)
    for k in range(cl_excess.shape[0]):
        cl_out[k] =  np.interp(lin, bc, cl_bin[k])

    return cl_out

def get_systematics(LR):

    # Filename
    fn = 'systematics_{:s}.pickle'.format(LR)

    # Load
    f = open(fn, 'rb')
    x = cP.load(f)
    f.close()

    # Get excess c_l
    cl_excess_217 = get_excess(x['clhm_217'], x['clffp8_217'])
    cl_excess_353 = get_excess(x['clhm_353'], x['clffp8_353'])
    cl_excess_cross = get_excess(x['clhm_cross'], x['clffp8_cross'])

    # Get systematics free R
    y = {'LR16':0.2, 'LR24':0.3, 'LR33':0.4, 'LR42':0.5, 'LR53':0.6, 
         'LR63N':0.7, 'LR63S':0.7, 'LR63':0.7, 'LR72':0.8}
    fsky = y[LR]
    mod = decorr.Model()
    RE,l = mod.getR(fsky, spec=1)
    RB,l = mod.getR(fsky, spec=2)

    # Add systematics
    syst = np.zeros( (3, 3, cl_excess_217[0].size) )
    syst[0,:,:] = cl_excess_217[0:3]/2
    syst[1,:,:] = cl_excess_353[0:3]/2
    #syst[2,:,:] = cl_excess_cross[0:3]

    REsyst,l = mod.getR(fsky, spec=1, syst=syst)
    RBsyst,l = mod.getR(fsky, spec=2, syst=syst)

    return syst, RE, RB, REsyst, RBsyst

    
