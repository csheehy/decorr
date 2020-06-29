import decorr
import numpy as np
import scipy.io
from matplotlib.pyplot import *

# A dust from BK14 analysis
A = 3.75

# Convert to fsky
I353 = decorr.A2I353(A)
fsky = decorr.I3532fsky(I353)


# Get E and B models
RmodE,lmod = decorr.getR(fsky,spec=1)
RmodB,lmod = decorr.getR(fsky,spec=2)

# Get simset points, saved from MATLAB
x = scipy.io.loadmat('decorr_simset.mat');

RB = x['RB']
RE = x['RE']
l = x['l']

# Get Planck real data (commented out for speed, having pre-loaded)
#REpl,bcE,RBpl,bcB = decorr.calccorr(0.6)

clf();

subplot(2,1,1)
semilogx(l,RE,'or');xlim(9,1000);ylim(.5,1.1);grid('on')
semilogx(bcE,REpl,'sk');
semilogx(lmod,RmodE,'r')
#legend(loc='lower left')
title('EE')
ylabel('$<217x353>/[<353x353><217x217>]^{1/2}$')

subplot(2,1,0)
semilogx(l,RB,'or',label='BK simset ("simd")');xlim(9,1000);ylim(0,1.4);grid('on')
semilogx(bcB,RBpl,'sk',label='Planck real maps $(f_{{sky}} = 0.6)$');
semilogx(xx,yy,'^b',label='Planck real maps, Aumont bins');
semilogx(lmod,RmodB,'r',label='$A_{dust} = 3.75 \mu K^2 (f_{{sky}} = 0.13)$')
title('BB')
ylabel('$<217x353>/[<353x353><217x217>]^{1/2}$')

# fsky = 0.22, .53, 0.7
A = [10, 50, 100]
linestyle = ['k--','k:','k-.']
for k,val in enumerate(A):
    fsky=decorr.I3532fsky(decorr.A2I353(val))
    RmodE,lmod = decorr.getR(fsky,spec=1)
    RmodB,lmod = decorr.getR(fsky,spec=2)
    leg = '$A_{{dust}} = {0} \mu K^2 (f_{{sky}} = {1:.2f})$'.format(val,np.float(fsky))
    subplot(2,1,1)
    semilogx(lmod,RmodE,linestyle[k])
    subplot(2,1,0)
    semilogx(lmod,RmodB,linestyle[k],label=leg)

legend(loc='lower left',fontsize=10)
xlabel('ell')

