import decorr
import numpy as np
from matplotlib.pyplot import *
ion()

doload=False
if doload:
    bt = 'lin'
    c1=decorr.Calc('spec/dust0_tophat/synch_therm_LR72_noihmds1_xxxx.pickle',bintype=bt)
    c2=decorr.Calc('spec/dust0_tophat/synch_therm_LR72_noihmds2_xxxx.pickle',bintype=bt)
    cavg=decorr.Calc('spec/dust0_tophat/synch_therm_LR72_noihmdsavg_xxxx.pickle',bintype=bt)
    c=decorr.Calc('spec/gaussian/synch_therm_LR72_qucov_noise_xpol_xxxx.pickle',bintype=bt)
    
close(1)
figure(1, figsize=(5,7))

clf()
yl=(.5,2)
xl=(20,300)


#################
subplot(2,1,1)
plot(c1.bc, c1.SN[:,2,:].T, color='gray')
errorbar(c1.bc,c1.R[2],c1.err[2],fmt='.k',label='hm1,ds2 x hm2,ds1', zorder=100)
xlim(*xl)
ylim(*yl)
grid('on')
setp(gca().get_xticklabels(), visible=False)
gca().set_yticks([.5,1.0,1.5])
legend(loc='upper left')
ylabel(r'$\mathcal{R}_{BB}$')

fig = gcf()
ax = gca()
pos = ax.get_position()
fig.add_axes([pos.x0+.15, pos.y0+.22, 0.25, 0.15])
plot(c1.bc, c1.SN[:,2,:].T, color='gray')
errorbar(c1.bc,c1.R[2],c1.err[2],fmt='.k', zorder=100)
xlim(19,90)
ylim(0.93,1.07)
gca().set_yticks([0.95,1.0,1.05])
gca().set_xticks([20,50,80])
grid('on')

#################
subplot(2,1,2)
plot(c2.bc, c2.SN[:,2,:].T, color='gray')
errorbar(c2.bc,c2.R[2],c2.err[2],fmt='.k',label='hm1,ds1 x hm2,ds2', zorder=100)
xlim(*xl)
ylim(*yl)
grid('on')
gca().set_yticks([.5,1.0,1.5])
legend(loc='upper left')
ylabel(r'$\mathcal{R}_{BB}$')
xlabel(r'Multipole $\ell$')

ax = gca()
pos = ax.get_position()
fig.add_axes([pos.x0+.15, pos.y0+.22, 0.25, 0.15])
plot(c2.bc, c2.SN[:,2,:].T, color='gray')
errorbar(c2.bc,c2.R[2],c2.err[2],fmt='.k',zorder=100)
xlim(19,90)
ylim(0.93,1.07)
gca().set_yticks([0.95,1.0,1.05])
gca().set_xticks([20,50,80])
grid('on')

tight_layout()
subplots_adjust(hspace=0, wspace=0)


#################
close(2)
figure(2, figsize=(6,6))

ysim = (c1.SN-c2.SN)[:,2,:]/2
yr = (c1.R-c2.R)[2]/2
yerr = np.nanstd(ysim,0)
plot(c1.bc, ysim.T, color='gray')
errorbar(c1.bc, yr, yerr, fmt='ok',zorder=400)
xlim(*xl)
ylim(-.4,.4)
grid('on')
legend(loc='upper left')
ylabel(r'$\Delta\mathcal{R}_{BB}/2$')

xlabel(r'Multipole $\ell$')
tight_layout()


#####################
close(3)
figure(3, figsize=(4,4))

ind = (c1.bc>=50) & (c1.bc<=160)
chi2sim = nansum((ysim[:,ind]/yerr[ind])**2,1)
chi2r = nansum((yr[ind]/yerr[ind])**2)
hist(chi2sim, color='gray')
yl=ylim()
plot([chi2r,chi2r],yl,'--k')
xlabel(r'$\chi^2 (50 \leq \ell \leq 160)$')
ylabel('N')
tight_layout()

