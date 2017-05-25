import decorr
import numpy as np
import cPickle as cP
from matplotlib.pyplot import *



class loaddata(object):

    def __init__(self):
        # Load data

        self.cqu = {}
        self.cmc = {}
        self.cd0 = {}
        self.cd2 = {}

        self.cqulin = {}
        self.cmclin = {}
        self.cquall = {}
        self.cmcall = {}

        self.cmcf = {}
        self.cd0f = {}
        self.cd2f = {}

        self.cmcfall = {}

        LR = decorr.getLR()

        for k,val in enumerate(LR):

            print(val)

            dir = 'spec/gaussian'
            d0dir = 'spec/dust0_tophat'
            d2dir = 'spec/dust2_tophat'

            fnqu = '{0}/synch_therm_{1}_qucov_noise_xxxx.pickle'.format(dir,val)
            fnmc = '{0}/synch_therm_{1}_mc_noise_xxxx.pickle'.format(dir,val)
            fnd0 = '{0}/synch_therm_{1}_mc_noise_xxxx.pickle'.format(d0dir,val)
            fnd2 = '{0}/synch_therm_{1}_mc_noise_xxxx.pickle'.format(d2dir,val)

            self.cqu[val] = decorr.Calc(fnqu, bintype='planck', full=False)
            self.cmc[val] = decorr.Calc(fnmc, bintype='planck', full=False)
            self.cd0[val] = decorr.Calc(fnd0, bintype='planck', full=False)
            self.cd2[val] = decorr.Calc(fnd2, bintype='planck', full=False)

            self.cqulin[val] = decorr.Calc(fnqu, bintype='lin', full=False, lmin=50, lmax=700)
            self.cmclin[val] = decorr.Calc(fnmc, bintype='lin', full=False, lmin=50, lmax=700)
            self.cquall[val] = decorr.Calc(fnqu, bintype='lin', full=False, lmin=50, lmax=700, nbin=1)
            self.cmcall[val] = decorr.Calc(fnmc, bintype='lin', full=False, lmin=50, lmax=700, nbin=1)

            self.cmcf[val] = decorr.Calc(fnmc, bintype='planck', full=True)
            self.cd0f[val] = decorr.Calc(fnd0, bintype='planck', full=True)
            self.cd2f[val] = decorr.Calc(fnd2, bintype='planck', full=True)
        
            self.cmcfall[val] = decorr.Calc(fnmc, bintype='lin', full=True, lmin=50, lmax=700, nbin=1)


class paper_plots(object):
    
    def __init__(self, c):
        self.c = c
        return


    def plotspec(self, LR):

        cqu = self.c.cqu[LR]
        cmc = self.c.cmc[LR]
        cqulin = self.c.cqulin[LR]
        cmclin = self.c.cmclin[LR]

        # Get model
        mod = decorr.Model(fsky=cqu.spec.fsky)

        spec = ['T', 'E','B']

        close(1)
        figure(1, figsize=(5,8))

        for k in [1,2]:
            subplot(2,1,k)

            xerr = (cqulin.be[1:] - cqulin.be[0:-1])/2
            errorbar(cqulin.bc, cqulin.R[k], yerr=cqulin.err[k], fmt=',',
                     linewidth=2, color='0.8') 


            plot(cqu.bc+8, cmc.SNmedian[k], 'xk', label='median of MC sims', markeredgewidth=2,zorder=100)
            plot(cqu.bc+16, cqu.SNmedian[k], 'xm', label='median of QUcov sims', markeredgewidth=2)
            plot(cqu.bc+24, cqu.Smedian[k], 'xb', label='med of noiseless sims', markeredgewidth=2)

            if k==1:
                plot(mod.l, mod.RE, 'k--', label='model')
            else:
                plot(mod.l, mod.RB, 'k--', label='model')

            xerr = (cqu.be[1:] - cqu.be[0:-1])/2
            errorbar(cqu.bc, cqu.R[k], yerr=cqu.err[k], xerr=xerr, fmt='o',
                     label='real', linewidth=2, color='r', ms=5.0, capthick=2) 


            plot([cqu.be[0],cqu.be[-1]],[1,1],':k')

            xlim(cqu.be[0], cqu.be[-1])
            if k==1:
                ylim(0.5,1.0)
                yticks(np.arange(0.5, 1.01, 0.1))
            else:
                ylim(0.7,1.1)
                yticks(np.arange(0.7, 1.11, 0.1))
            
            title(spec[k])
            if k==1:
                legend()
                


            if k==2:
                xlabel(r'Multipole $\ell$')
                ylabel(r'$\mathcal{R}_{\ell}^{BB}$')
            else:
                ylabel(r'$\mathcal{R}_{\ell}^{EE}$')
                
        tight_layout()


    def plotnh(self):

        # Massage into proper form
        LRin = self.c.cmc.keys()
        nhin = {'LR16':1.32, 'LR24':1.65, 'LR33':2.12, 'LR42':2.69, 'LR53':3.45, 
              'LR63':4.41, 'LR63N':4.14, 'LR63S':4.70, 'LR72':6.02}


        # Sort LR by increasing nh
        nhh = nhin.values()
        ind = np.argsort(nhh)
        LRR = nhin.keys()
        LR = []
        nh = []
        for k,val in enumerate(ind):
            if LRR[val] in LRin:
                LR.append(LRR[val])
                nh.append(nhin[LRR[val]])
        nh = np.array(nh)

        # Get data
        R = []
        SNqu = []
        SNmc = []
        for k,val in enumerate(LR):
            R.append(self.c.cmc[val].R[2,0])
            SNqu.append(self.c.cqu[val].SN[:,2,0])
            SNmc.append(self.c.cmc[val].SN[:,2,0])
        R = np.array(R)
        SNqu = np.array(SNqu)
        SNmc = np.array(SNmc)


        col = ['y','b']
        
        #################
        #close(2)
        figure(2, figsize=(7,5))
        clf()
        for jj in range(2):
            # QUcov noise, then mc_noise

            med = []
            up68 = []
            down68 = []
            up95 = []
            down95 = []

            for k,val in enumerate(LR):

                # SN sims
                if jj==0:
                    x = SNqu[k]
                else:
                    x = SNmc[k]

                # Confidence intervals
                xmed = np.nanmedian(x)
                ngood = np.size(x[np.isfinite(x)])

                ntry = 1000
                tryvals = (np.nanmax(x) - np.nanmin(x))
                dx = np.linspace(0, tryvals, ntry)
                nupper = np.zeros(ntry)
                nlower = np.zeros(ntry)
                for j,val in enumerate(dx):
                    nupper[j] = np.where(x[x<(xmed+val)])[0].size
                    nlower[j] = np.where(x[x>(xmed-val)])[0].size

                up68.append(np.interp(0.5 + 0.68/2, nupper*1.0/ngood, dx))
                up95.append(np.interp(0.5 + 0.95/2, nupper*1.0/ngood, dx))
                down68.append(np.interp(0.5 + 0.68/2, nlower*1.0/ngood, dx))
                down95.append(np.interp(0.5 + 0.95/2, nlower*1.0/ngood, dx))
                med.append(xmed)
                

            wd = 0.1
            for k,val in enumerate(med):
                left = nh[k] + wd*(jj-1) 
                bottom68 = med[k] - down68[k]
                bottom95 = med[k] - down95[k]
                height68 = down68[k] + up68[k]
                height95 = down95[k] + up95[k]

                lab68 = None
                lab95 = None
                labmed = None
                labmed = None

                if k==0 and jj==0:
                    labmed= 'median of QUcov sims'
                    lab68 = '68% C.L.'
                    lab95 = '95% C.L.'

                if k==0 and jj==1:
                    labmed = 'median of MC sims'

                bar(left, height95, width=wd, bottom=bottom95, color='0.8',
                    edgecolor='0.4', label=lab95)
                bar(left, height68, width=wd, bottom=bottom68, color='0.5',
                    edgecolor='0.4', label=lab68)
                plot([left,left+wd],[med[k], med[k]],col[jj], linewidth=2, label=labmed)

        ax = gca()
        handles, labels = ax.get_legend_handles_labels()

        # Plot a few realizations
        h1 = plot(nh, SNmc[:, 0:10], color='m', alpha=0.3)
        h2 = plot(nh, R, 'Dr', markersize=8.0, markeredgewidth=1.0)
        plot([0,7],[1,1],'k:')
        xlabel(r'$N_H/10^{20} [cm^{-2}]$')
        ylabel(r'$\mathcal{R}_{50-160}^{BB}$')
        xlim(1,7)
        ylim(0.8,1.3)

        handles.append(h1[0])
        handles.append(h2[0])
        labels.append(u'QUcov sims')
        labels.append(u'real')
        
        legend(handles, labels, loc='upper right')
        tight_layout()


        ############################
        ############################
        # Get some statistics
        chiR = np.sum(R-1)
        chi2R = np.sum((R-1)**2)

        chiSNqu = np.sum(SNqu-1, axis=0)
        chi2SNqu = np.sum((SNqu-1)**2, axis=0)

        chiSNmc = np.sum(SNmc-1, axis=0)
        chi2SNmc = np.sum((SNmc-1)**2, axis=0)

        nrlz = chi2SNqu.size*1.0
        PTEchi2qu = np.size(np.where(chi2SNqu > chi2R)[0]) / nrlz
        PTEchi2mc = np.size(np.where(chi2SNmc > chi2R)[0]) / nrlz
        print('chi2 PTE (QU) = {0}'.format(PTEchi2qu))
        print('chi2 PTE (MC) = {0}'.format(PTEchi2mc))

        PTEchiqu = np.size(np.where(chiSNqu > chiR)[0]) / nrlz
        PTEchimc = np.size(np.where(chiSNmc > chiR)[0]) / nrlz
        print('chi PTE (QU) = {0}'.format(PTEchiqu))
        print('chi PTE (MC) = {0}'.format(PTEchimc))


        #close(3)
        figure(3, figsize=(7,5))
        clf()

        subplot(1,2,1)
        hist(chi2SNmc, range=(0,0.2), bins=15, label='MC')
        hist(chi2SNqu, range=(0,0.2), bins=15, alpha=0.5, color='r', label='QUcov')
        yl=ylim()
        plot( [chi2R,chi2R], [0,yl[1]], 'r', linewidth=3, label='real')
        ylim(0.1,yl[1])
        xlim(0,0.2)
        xlabel(r'$\sum (\mathcal{R}_{50-160}^{BB}-1)^2$')
        ax=gca()
        text(0.95, 0.98, 'PTE (MC) = {:0.2f}'.format(PTEchi2mc),
             horizontalalignment='right', verticalalignment='top', 
             transform=ax.transAxes)
        text(0.95, 0.94, 'PTE (QUcov) = {:0.2f}'.format(PTEchi2qu),
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)

        legend(loc=(0.58,0.72))

        subplot(1,2,2)
        hist(chiSNmc, range=(-1,1), bins=15, label='MC')
        hist(chiSNqu, range=(-1,1), bins=15, alpha=0.5, color='r', label='QUcov')
        yl=ylim()
        plot( [chiR,chiR], [0,yl[1]], 'r', linewidth=3, label='real')
        ylim(0.1,yl[1])
        xlabel(r'$\sum (\mathcal{R}_{50-160}^{BB}-1)$')
        tight_layout()

        ax=gca()
        text(0.95, 0.98, 'PTE (MC) = {:0.2f}'.format(PTEchimc),
             horizontalalignment='right', verticalalignment='top', 
             transform=ax.transAxes)
        text(0.95, 0.94, 'PTE (QUcov) = {:0.2f}'.format(PTEchiqu),
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
        tight_layout()

