import decorr
import numpy as np
import cPickle as cP
from matplotlib.pyplot import *
from matplotlib import gridspec
from scipy import convolve
from scipy.stats import binom
from copy import deepcopy as dc

def nanhist(x, **kwargs):
    hist(x[np.isfinite(x)],**kwargs)

class loaddata(object):

    def __init__(self, full=False, db=False, Rt='R'):
        # Load data

        self.cqu = {}
        self.cmc = {}
        self.cd0 = {}
        self.cd2 = {}

        self.cqulin = {}
        self.cmclin = {}

        self.cquall = {}
        self.cmcall = {}

        self.cd0lin = {}
        self.cd1lin = {}
        self.cd2lin = {}
        self.cd1bk = {}
        self.cd2bk = {}
        self.cmcbk = {}


        LR = decorr.getLR()

        print('full = {0}, dodebias = {1}, Rtype={2}'.format(full,db,Rt))

        for k,val in enumerate(LR):

            print(val)

            dir = 'spec/gaussian'
            d0dir = 'spec/dust0_tophat'
            d1dir = 'spec/dust1_tophat'
            d2dir = 'spec/dust2_tophat'

            fnqu = '{0}/synch_therm_{1}_qucov_noise_xxxx.pickle'.format(dir,val)
            fnmc = '{0}/synch_therm_{1}_mcplusexcess_noise_xxxx.pickle'.format(dir,val)
            fnd0 = '{0}/synch_therm_{1}_mcplusexcess_noise_xxxx.pickle'.format(d0dir,val)
            fnd1 = '{0}/synch_therm_{1}_mcplusexcess_noise_xxxx.pickle'.format(d1dir,val)
            fnd2 = '{0}/synch_therm_{1}_mcplusexcess_noise_xxxx.pickle'.format(d2dir,val)

            self.cqu[val] = decorr.Calc(fnqu, bintype='planck', full=full, dodebias=db, Rtype=Rt)
            self.cmc[val] = decorr.Calc(fnmc, bintype='planck', full=full, dodebias=db, Rtype=Rt)
            #self.cd0[val] = decorr.Calc(fnd0, bintype='planck', full=full, dodebias=db, Rtype=Rt)
            #self.cd2[val] = decorr.Calc(fnd2, bintype='planck', full=full, dodebias=db, Rtype=Rt)

            self.cqulin[val] = decorr.Calc(fnqu, bintype='lin', full=full, lmin=0, lmax=700, nbin=70, dodebias=db, Rtype=Rt)
            self.cmclin[val] = decorr.Calc(fnmc, bintype='lin', full=full, lmin=0, lmax=700, nbin=70, dodebias=db, Rtype=Rt)

            #self.cquall[val] = decorr.Calc(fnqu, bintype='lin', full=full, lmin=50, lmax=700, nbin=1, dodebias=db, Rtype=Rt)
            #self.cmcall[val] = decorr.Calc(fnmc, bintype='lin', full=full, lmin=50, lmax=700, nbin=1, dodebias=db, Rtype=Rt)

            #self.cd0lin[val] = decorr.Calc(fnd0, bintype='lin', lmin=0, lmax=700, nbin=70, full=full, dodebias=db, Rtype=Rt)
            self.cd1lin[val] = decorr.Calc(fnd1, bintype='lin', lmin=0, lmax=700, nbin=70, full=full, dodebias=db, Rtype=Rt)
            self.cd2lin[val] = decorr.Calc(fnd2, bintype='lin', lmin=0, lmax=700, nbin=70, full=full, dodebias=db, Rtype=Rt)
            #self.cd1bk[val] = decorr.Calc(fnd1, bintype='bk', full=full, dodebias=db, Rtype=Rt)
            #self.cd2bk[val] = decorr.Calc(fnd2, bintype='bk', full=full, dodebias=db, Rtype=Rt)
            self.cmcbk[val] = decorr.Calc(fnmc, bintype='bk', full=full, dodebias=db, Rtype=Rt)


class paper_plots(object):
    
    def __init__(self, c):
        """Plots are:
        plotspec() - plot binned R_BB and R_EE spectra
        plotnh() - plot R_BB^50-160 as function of nh and the histogram of statistics derived from this
        plotrawspec() - plot raw dust spectra to make sure they work
        plotnoispec() - plot noise spectra
        """
        self.c = c
        return

    def plotall(self):
        """Make all plots"""
        self.plotspec('LR63')
        self.plotnh('QUMC', bin=0)
        self.plotnh('QUMC', bin='all')
        self.plotnh('d0d2', bin=0)
        self.plotrawspec()
        self.plotnoispec()
        self.PTEtable()
        self.MCPTE()


    def plotspec(self, LR='LR63'):

        cqu = self.c.cqu[LR]
        cmc = self.c.cmc[LR]
        cqulin = self.c.cqulin[LR]
        cmclin = self.c.cmclin[LR]

        # Get model
        mod = decorr.Model(fsky=cqu.spec.fsky)
        modbin = decorr.Model(fsky=cqu.spec.fsky, be=cqu.be)

        spec = ['TT', 'EE','BB']


        for k in [1,2]:

            close(k)
            figure(k, figsize=(7,5))


            xerr = (cqulin.be[1:] - cqulin.be[0:-1])/2
            errorbar(cmclin.bc, cmclin.R[k], yerr=cmclin.err[k], fmt=',',
                     linewidth=2, color=[0.8,0.8,0.8]) 


            #plot(cqu.bc+8, cmc.SNmedian[k], 'xk', label='median of MC sims', markeredgewidth=2,zorder=100)
            #plot(cqu.bc+16, cqu.SNmedian[k], 'xm', label='median of QU sims', markeredgewidth=2)
            plot(cqu.bc-4, cmc.Smean[k], '.b', label='mean of noiseless sims')

            if k==1:
                y = mod.RE
                ybin = modbin.RE
                x = np.loadtxt('PlanckL_EE.csv',delimiter=',')
            else:
                y = mod.RB
                ybin = modbin.RB
                x = np.loadtxt('PlanckL_BB.csv',delimiter=',')

            # Plot binned model
            plot(cqu.bc+6, ybin, '+m', label='binned model',zorder=100,mew=2)

            # Plot model
            plot(mod.l, y, 'k--', label='model')

            ## Plot Planck data
            #if LR == 'LR63':
             #   plot(cqu.bc, x[:,1], 'bs', label='Planck L',)

            # Plot real data
            xerr = (cmc.be[1:] - cmc.be[0:-1])/2
            errorbar(cqu.bc-4, cqu.R[k], yerr=cqu.err[k], xerr=xerr, fmt='o',
                     label='data', linewidth=1, color=[0.8,0.2,0.2], ms=2.0, capthick=1, capsize=2) 
            errorbar(cmc.bc, cmc.R[k], yerr=cqu.err[k], xerr=xerr, fmt='o',
                     label='data after debias', linewidth=2, color='k', ms=4.0, capthick=1, capsize=3) 


            plot([cqu.be[0],cqu.be[-1]],[1,1],':k')

            xlim(cqu.be[0], cqu.be[-1])
            if k==1:
                ylim(0.4,1.4)
                #yticks(np.arange(0.5, 1.01, 0.1))
            else:
                ylim(0.7,1.2)
                #yticks(np.arange(0.7, 1.11, 0.1))
            
            title(spec[k])
            if k==1:
                legend(loc='upper left')
                

            xlabel(r'Multipole $\ell$')
            if k==2:
                ylabel(r'$\mathcal{R}_{\ell}^{BB}$')
            else:
                ylabel(r'$\mathcal{R}_{\ell}^{EE}$')
                
            tight_layout()

        figure(1)
        savefig('figures/{0}spec_EE.pdf'.format(LR))

        figure(2)
        savefig('figures/{0}spec_BB.pdf'.format(LR))
        

    def plotnh(self, bin=0, type=''):

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
        Rqu = []
        Rmc = []
        SN = []
        err = []

        for k,val in enumerate(LR):
            if type == '':
                Rqu.append(self.c.cqu[val].R[2,bin])
            Rmc.append(getattr(self.c,'cmc'+type)[val].R[2,bin])
            SN.append(getattr(self.c,'cmc'+type)[val].SN[:,2,bin])
            err.append(getattr(self.c,'cmc'+type)[val].err[2,bin])


        if type=='':
            Rqu = np.array(Rqu)
        Rmc = np.array(Rmc)
        SN = np.array(SN).T
        err = np.array(err)

        be = getattr(self.c,'cmc'+type)[val].be[bin:(bin+2)].astype(int)
        binlab = '{:d}-{:d}'.format(*be)


        med = []
        up68 = []
        down68 = []
        up95 = []
        down95 = []

        for k,val in enumerate(LR):

            x = SN[:,k]

            # Confidence intervals
            xmed = np.nanmedian(x)

            med.append(xmed)
            ind = np.isfinite(x)
            up68.append(np.percentile(x[ind], 50 + 68./2))
            up95.append(np.percentile(x[ind], 50 + 95./2))
            down68.append(np.percentile(x[ind], 50 - 68./2))
            down95.append(np.percentile(x[ind], 50 - 95./2))

        col = ['y','b']

        #################
        close(1)
        figure(1, figsize=(7,5))
        clf()

        wd = 0.1
        for k,val in enumerate(med):
            left = nh[k] - wd/2
            bottom68 = down68[k]
            bottom95 = down95[k]
            height68 = up68[k] - down68[k]
            height95 = up95[k] - down95[k]

            lab68 = None
            lab95 = None
            labmed = None
            labmed = None

            if k==0:
                labmed= 'median of sims'
                lab68 = '68% C.L.'
                lab95 = '95% C.L.'

            if k==0:
                labmed = 'median of sims'

            bar(left, height95, width=wd, bottom=bottom95, color='0.8',
                edgecolor='0.4', label=lab95)
            bar(left, height68, width=wd, bottom=bottom68, color='0.5',
                edgecolor='0.4', label=lab68)
            plot([left,left+wd],[med[k], med[k]], 'b', linewidth=2, label=labmed)

        ax = gca()
        handles, labels = ax.get_legend_handles_labels()

        # Plot a few realizations
        nplots=10
        colormap = cm.nipy_spectral 
        ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1, nplots)])
        h1 = plot(nh, SN[0:nplots].T, alpha=0.3)

        # Plot real
        hmc = plot(nh, Rmc, 'Dr', markersize=4.0, markeredgewidth=1.0)
        if type == '':
            hqu = plot(nh, Rqu, '.')
        plot([0,7],[1,1],'k:')
        xlabel(r'$N_H/10^{20} [cm^{-2}]$')
        ylabel(r'$\mathcal{R}_{' + binlab + r'}^{BB}$')
        xlim(1,7)
        ylim(0.7,1.3)

        # Plot planck data
        #if (bin==0) & (type==''):
        #    x = np.loadtxt('PlanckL_NH.csv', delimiter=',')
        #    h3 = plot(nh, x[:,1], 's', markersize=2.0)

        handles.append(h1[0])
        labels.append(u'sims')        

        #if (bin==0) & (type==''):
        #    handles.append(h3[0])
        #    labels.append(u'Planck L')

        if type == '':
            handles.append(hqu[0])
            labels.append(u'data')

        handles.append(hmc[0])
        labels.append(u'data after debias')

        legend(handles, labels, loc='upper right')
        tight_layout()


        figure(1)
        savefig('figures/nh_bin{:s}_curve.pdf'.format(binlab))

        if type != '':
            return

        ############################
        ############################
        # Get some statistics
        err = err**2

        chiRmc = np.sum((Rmc-1)/err)
        chi2Rmc = np.sum(((Rmc-1)/err)**2)
        chiRqu = np.sum((Rqu-1)/err)
        chi2Rqu = np.sum(((Rqu-1)/err)**2)

        chiSN = np.sum((SN-1)/err, axis=1)
        chi2SN = np.sum(((SN-1)/err)**2, axis=1)

        nrlz = chi2SN.size*1.0
        PTEchi2qu = np.size(np.where(chi2SN < chi2Rqu)[0]) / nrlz
        PTEchi2mc = np.size(np.where(chi2SN < chi2Rmc)[0]) / nrlz
        print('chi2 PTE = {0}'.format(PTEchi2qu))
        print('chi2 PTE after debias = {0}'.format(PTEchi2mc))

        PTEchiqu = np.size(np.where(chiSN < chiRqu)[0]) / nrlz
        PTEchimc = np.size(np.where(chiSN < chiRmc)[0]) / nrlz
        print('chi PTE = {0}'.format(PTEchiqu))
        print('chi PTE after debias = {0}'.format(PTEchimc))

        # Calculate number of zero crossings
        def nzero(x):
            if np.ndim(x) == 1:
                x = np.reshape(x, [1, x.size])
            sgx = np.sign(x-1)
            cross = (sgx[:,1:] != sgx[:,0:-1]).astype(float)
            ncross = np.sum(cross, axis=1)
            return ncross

        ncross = nzero(SN)
        ncrossr = nzero(Rmc)

        ############################
        ############################

        


        close(2)
        figure(2, figsize=(7,5))
        clf()

        subplot(1,2,1)
        rng = (0, np.max(np.hstack((chi2SN, chi2SN))))
        hist(chi2SN, range=rng, bins=25, label='sims')
        yl=ylim()
        plot( [chi2Rqu,chi2Rqu], [0,yl[1]], 'orange', linewidth=3, label='data')
        plot( [chi2Rmc,chi2Rmc], [0,yl[1]], 'r', linewidth=3, label='data after debias')
        ylim(0.1,yl[1])
        xlabel(r'$\chi^2$')
        ax=gca()
        ax.xaxis.major.locator.set_params(nbins=4) 

        text(0.95, 0.98, 'PTE = {:0.2f}'.format(PTEchi2qu),
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)
        text(0.95, 0.94, 'PTE after debias = {:0.2f}'.format(PTEchi2mc),
             horizontalalignment='right', verticalalignment='top', 
             transform=ax.transAxes)

        legend(loc=(0.25,0.72))

        subplot(1,2,2)
        rng = ( np.min(np.hstack((chiSN, chiSN))), np.max(np.hstack((chiSN, chiSN))) )
        hist(chiSN, range=rng, bins=15, label='sims')
        yl=ylim()
        plot( [chiRqu,chiRqu], [0,yl[1]], 'orange', linewidth=3, label='data')
        plot( [chiRmc,chiRmc], [0,yl[1]], 'r', linewidth=3, label='data after debias')
        ylim(0.1,yl[1])
        xlabel(r'$\chi$')

        tight_layout()

        ax=gca()
        ax.xaxis.major.locator.set_params(nbins=4)

        text(0.95, 0.98, 'PTE = {:0.2f}'.format(PTEchiqu),
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes)

        text(0.95, 0.94, 'PTE after debias = {:0.2f}'.format(PTEchimc),
             horizontalalignment='right', verticalalignment='top', 
             transform=ax.transAxes)

        tight_layout()

        ####################
        # Plot histogram of zero crossings
        close(3)
        figure(3, figsize=(7,5))

        N, be, dum = hist(ncross, range=(-0.5,8.5), bins=9, label='sims')
        yl=ylim()
        plot( [ncrossr,ncrossr], [0,yl[1]], 'orange', linewidth=3, label='data')
        xlim(-0.5,8.5)
        xlabel(r'$\mathcal{R^{\mathrm{BB}}}$, # of zero crossings') 

        # Binomial distribution
        N = np.arange(9)
        bn = binom(8, 0.5)
        Nexp = bn.pmf(N)*nrlz
        for k,val in enumerate(Nexp):
            if k==0:
                lab = 'uncorr. exp.'
            else:
                lab = None
            plot([be[k],be[k+1]], [val,val], '--k', linewidth=2, label=lab)
        legend()

        ##################
        # Save figs

        figure(2)
        savefig('figures/nh_bin{:s}_dist.pdf'.format(binlab))

        figure(3)
        savefig('figures/nh_bin{:s}_crossings.pdf'.format(binlab))


    def PTEtable(self):

        LR = decorr.getLR()
        # Reorder
        LR = np.array(LR)[ [0,1,2,3,4,5,7,6,8] ]

        f = open('figures/PTEtable.tex','w')
        
        
        f.write(r'\begin{table*}[tbp!] '+'\n')
        f.write(r'\centering '+'\n')
        f.write(r'\caption{The caption} '+'\n')
        f.write(r'\label{tab:ptes} '+'\n')
        f.write(r'\begin{tabular}{lrrrrrrrrrr} '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\rule{0pt}{2ex} '+'\n')
        f.write(r' & & \mc{LR16}& \mc{LR24}& \mc{LR33}& \mc{LR42}& \mc{LR53}& \mc{LR63N}& \mc{LR63}& \mc{LR63S}& \mc{LR72} \\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\multispan2$f_{\rm sky}^{\rm eff}$ [\%]\hfil& 16& 24& 33& 42& 53& 33& 63& 30& 72\\ '+'\n')
        f.write(r'\multispan2$N_{\mathrm{HI}}$ [$10^{20}\,{\rm cm}^{-2}$]\hfil& 1.32& 1.65& 2.12& 2.69& 3.45& 4.14& 4.41& 4.70& 6.02\\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r' & \, $\ell$ range \hfill &\\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')

        #for n in [0,1,2,3]:
        for n in [0,1]:

            if n in [0,1]:
                spec = 2 # BB
                speclab = 'BB'
            else:
                spec = 1 # EE
                speclab = 'EE'

            if n in [0,2]:
                noilab = 'MC'
                attr = 'cmc'
            else:
                noilab = 'QU'
                attr = 'cqu'

            ##########
            f.write(r'\multispan1$\rm{'+speclab+r'}\ \rm{PTE}_{\rm{'+noilab+r'}}$')

            bins = ['50--700','50--160','160--320','320--500','500--700']
            doind = [0,0,1,2,3]

            for j, binval in enumerate(bins):
                f.write(r' &{0} '.format(binval))
                for k, val in enumerate(LR):

                    if j==0:
                        x = 100 * getattr(self.c, attr)[val].PTEall[spec]
                    else:
                        x = 100 * getattr(self.c, attr)[val].PTE[spec,doind[j]]

                    if np.isfinite(x):
                        f.write(r'& {:0.1f}'.format(x))
                    else:
                        f.write(r'&\ldots')

                f.write(r'\\ '+'\n')

            f.write(r'\toprule '+'\n')
            if n==1:
                f.write(r'\toprule '+'\n')
            f.write(r'\addlinespace[1ex] '+'\n')


        f.write(r'\end{tabular} '+'\n')
        f.write(r'\end{table*} '+'\n')

        f.close()


    def plotrawspec(self):

        close(1)
        fig = figure(1, figsize=(10,8))

        LR = decorr.getLR()
        for k,val in enumerate(LR):
            subplot(3,3,k+1)
            ax = gca()

            l = self.c.cmclin[val].bc
            snmc = self.c.cmclin[val].sn[:,:,2,:].mean(0)
            yerr = self.c.cmclin[val].sn[:,:,2,:].std(0)
            r = self.c.cmclin[val].r[:,2,:]

            fac = 1e12

            ax.set_xscale("log", nonposx='clip')
            ax.set_yscale("log", nonposy='clip')
            errorbar(l, r[0]*fac, yerr[0]*fac, fmt='.b')
            plot(l, snmc[0]*fac, color=[.5,.5,1])

            errorbar(l+0.02*l, r[1]*fac, yerr[1]*fac, fmt='.r')
            plot(l+0.02*l, snmc[1]*fac, color=[1,.5,.5])

            errorbar(l+0.04*l, r[2]*fac, yerr[2]*fac, fmt='.g')
            plot(l+.04*l, snmc[2]*fac, color=[.5,1,.5])

            grid('on')
            xlim(20,700)
            ylim(1e-1, 5e2)
            ax = gca()
            ax.set_yticks([1e0,1e1,1e2])
            ax.set_xticks([50,100,500])
            ax.set_xticklabels(['50','100','500'])
            

            # Plot PIPL bins
            be = [50,160,320,500,700]
            yl = ylim()
            for j, val2 in enumerate(be):
                plot([val2,val2],[yl[0],yl[1]],'k--')

            text(0.97, 0.96, val,
                 horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes)

            if (k+1)<7:
                setp(ax.get_xticklabels(), visible=False)
            if (k+1) not in [1,4,7]:
                setp(ax.get_yticklabels(), visible=False)

            if (k+1) == 7:
                text(0.03, 0.3, r'$<353 \mathrm{x} 353>$', color='r',
                     horizontalalignment='left', verticalalignment='top',
                     transform=ax.transAxes)
                text(0.03, 0.2, r'$<217 \mathrm{x} 353>$', color='g',
                     horizontalalignment='left', verticalalignment='top',
                     transform=ax.transAxes)
                text(0.03, 0.1, r'$<217 \mathrm{x} 217>$', color='b',
                     horizontalalignment='left', verticalalignment='top',
                     transform=ax.transAxes)
                     

        subplots_adjust(hspace=0, wspace=0)
        
        ax = fig.add_axes( [0., 0., 1, 1] )
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(.05, 0.5, r'$\ell(\ell+1)\mathcal{C}_{\ell}^{\mathrm{BB}}/2\pi \ [\mu\mathrm{K}^2]$', 
                 rotation='vertical', horizontalalignment='center', verticalalignment='center')
        ax.text(.5, 0.05, r'Multipole $\ell$',
                 horizontalalignment='center', verticalalignment='center')


        savefig('figures/rawspec.pdf')



    def plotnoispec(self, LR='LR63'):


        nrlz =  self.c.cmclog[LR].nrlz
        l = self.c.cmclog[LR].bc
        fac = l*(l+1)/(2*np.pi)

        nqu = fac * self.c.cqulog[LR].n.mean(0)*1e12
        nmc = fac * self.c.cmclog[LR].n.mean(0)*1e12
        nquerr = fac * self.c.cqulog[LR].n.std(0) / np.sqrt(nrlz) * 1e12
        nmcerr = fac * self.c.cmclog[LR].n.std(0) / np.sqrt(nrlz) * 1e12

        close(1)
        f = figure(1, figsize=(5,9))
        xt = [100,500]

        sp = 2 # 1/2 -> EE/BB

        yl = [[-0.3,0.3], [-3,3], [-.3,.3]]


        # c = 0,1,2 -> <217x217> / <353x353> / <217x353>
        for c in [0,1,2]:

            if c==0:
                tit = r'$<217 \mathrm{x} 217>$'
            if c==1:
                tit = r'$<353 \mathrm{x} 353>$'
            if c==2:
                tit = r'$<217 \mathrm{x} 353>$'

            ymc = nmc[c,sp,:]
            yqu = nqu[c,sp,:]
            dymc = nmcerr[c,sp,:]
            dyqu = nquerr[c,sp,:]

            subplot(3,1,c+1)

            semilogx(l, yqu, 'r', label='QU')
            fill_between(l, yqu-dyqu, yqu+dyqu, color='r', alpha=0.5)

            semilogx(l, ymc, 'b', label='MC')
            fill_between(l, ymc-dymc, ymc+dymc, color='b', alpha=0.5)

            semilogx([50,700],[0,0],'k--',linewidth=2)


            xlim(50,700)
            #grid('on')
            ax = gca()
            ax.set_xticks(xt)
            ax.set_xticklabels(xt)

            ylabel(r'$\ell(\ell+1)\mathcal{C}_{\ell,\mathrm{Noise}}^{BB}/2\pi \ [\mu \mathrm{K}^2]$')
            if c==0:
                legend(loc='upper left')
            if c==2:
                xlabel(r'Multipole $\ell$')
            else:
                setp(ax.get_xticklabels(), visible=False)
            title(tit)
            ylim(*yl[c])
            #ax.set_yticks(ax.get_yticks()[1:])

            #ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        tight_layout()
        savefig('figures/noispec.pdf'.format(c))


    def MCPTE(self):
       """Compute PTE table on a realization by realization basis"""
       c = dc(self.c.cmc)

       LR = decorr.getLR()
       nLR = len(LR)
       nrlz = c[LR[0]].nrlz

       # Get real PTE table
       PTER =np.zeros(nLR*5)
       for j,val in enumerate(LR):

           PTER[j+0] = c[val].PTEall[2]
           PTER[j+9] = c[val].PTE[2,0]
           PTER[j+18] = c[val].PTE[2,1]
           PTER[j+27] = c[val].PTE[2,2]
           PTER[j+36] = c[val].PTE[2,3]
       
       # Get simulated PTE tables
       PTEtab = np.zeros((nLR*5, nrlz))
       for k in range(nrlz):
           for j,val in enumerate(LR):

               c[val].R = c[val].SN[k] # Replace real with SN realization
               c[val].getPTE()

               PTEtab[j+0 , k] = c[val].PTEall[2]
               PTEtab[j+9 , k] = c[val].PTE[2,0]
               PTEtab[j+18 , k] = c[val].PTE[2,1]
               PTEtab[j+27 , k] = c[val].PTE[2,2]
               PTEtab[j+36 , k] = c[val].PTE[2,3]
               

       PTEthresh = 0.01
       nsim = (PTEtab<=PTEthresh).astype(float).sum(0)
       nr = len(np.where(PTER<=PTEthresh)[0])

       nsimthresh = len(np.where(nsim>=nr)[0])

       # Get expectation value
       bn = binom(nLR*5, PTEthresh)
       nexp = (bn.pmf(np.arange(nr,100))).sum() * nrlz
       

       print('We observe {:d} PTEs <= {:0.1f}%.'.format(nr, PTEthresh*100))
       print('From a random dist. of uncorrelated PTEs, this occurence has a probability of {:0.1f}%.'.format(nexp/nrlz*100))
       print('From sims, this occurence has a probability of {:0.1f}%.'.format(nsimthresh*100./nrlz))




       # Plot correlation matrix
       close(1)
       fig = figure(1, figsize=(6,5))

       # Correlation matrix of SN R values
       Rtab = np.zeros((nLR*4, nrlz))
       for j,val in enumerate(LR):
           Rtab[j+np.array([0,9,18,27]), :] = c[val].SN[:,2,:].T

       # Choose which quantity to get correlation matrix of
       #dotab = PTEtab
       dotab = Rtab

       # Replace NaN's with uncorrelated uniformly distributed random numbers
       ind = np.where(~np.isfinite(dotab))
       dotab[ind] = np.random.rand(ind[0].size)

       # Calculate correlation matrix
       corr = np.corrcoef(dotab)

       imshow(corr)
       ax = gca()
       ax.set_aspect('equal')
       clim(-1,1)
       colorbar()

       yticks(np.arange(9*4), np.hstack((LR,LR,LR,LR)), fontweight='bold')
       tick_params(labelsize=6)

       xticks(np.arange(9*4), np.hstack((LR,LR,LR,LR)), fontweight='bold', rotation=90)
       tick_params(labelsize=6)
       
       yl = ylim()
       xl = xlim()

       ax = fig.add_axes( ax.get_position() )
       ax.set_axis_bgcolor('none')
       ax.set_xlim(xl)
       ax.set_ylim(yl)
       ax.set_aspect('equal')
       ax.set_xticks([])

       yticks(np.array([9,18,27,36])-5, 
              ['$50-160$','$160-320$','$320-500$','$500-700$'], rotation=90, multialignment='center')
       tick_params(pad=30, labelsize=10)

       xticks(np.array([9,18,27,36])-5, 
              ['$50-160$','$160-320$','$320-500$','$500-700$'], rotation=0, multialignment='center')
       tick_params(pad=30, labelsize=10)


       savefig('figures/corr_matrix.pdf')


       return PTEtab


    def plotRfine(self):
        """Plot R in fine bins"""
        
        val = 'LR72'

        bclin = self.c.cmclin[val].bc
        bcpipl = self.c.cmc[val].bc
        bcbk = self.c.cmcbk[val].bc

        belin = self.c.cmclin[val].be
        bepipl = self.c.cmc[val].be
        bebk = self.c.cmcbk[val].be

        Rlin = self.c.cmclin[val].R
        Rpipl = self.c.cmc[val].R
        Rbk = self.c.cmcbk[val].R

        errlin = self.c.cmclin[val].err
        errpipl = self.c.cmc[val].err
        errbk = self.c.cmcbk[val].err

        Sd2lin = self.c.cd2lin[val].S.mean(0)
        Sd1lin = self.c.cd1lin[val].S.mean(0)
        
        clf()

        for k in range(2):

            if k==1:
                fig = gcf()
                ax = gca()
                pos = ax.get_position()
                fig.add_axes([pos.x0+.07, pos.y0+.04, 0.25, 0.3])

            xerrlin = (belin[1:] - belin[0:-1])/2
            xerrpipl = (bepipl[1:] - bepipl[0:-1])/2
            xerrbk = (bebk[1:] - bebk[0:-1])/2

            if k==0:
                ylim(0.65,1.25)
                xlim(20,300)
                
            else:
                ylim(0.97,1.02)
                xlim(20,125)
                ax = gca()
                ax.set_yticks([0.97,1,1.02])

            errorbar(bcpipl, Rpipl[2], errpipl[2], xerrpipl, fmt='sr',markersize=4)
            errorbar(bclin, Rlin[2], errlin[2], xerrlin, fmt='.',color=[0.5,0.5,0.5])
            errorbar(bcbk, Rbk[2], errbk[2], xerrbk, fmt='ob', markersize=4)
            
            plot(bclin, Sd2lin[2], '--k')
            plot(bclin, Sd1lin[2], ':k')


    def Rlimittable(self):
        """Plot R lower limits or detections"""

        LR = decorr.getLR()
        # Reorder
        LR = np.array(LR)[ [0,1,2,3,4,5,7,6,8] ]


        f = open('figures/Rlimittable.tex','w')
        
        
        f.write(r'\begin{table*}[tbp!] '+'\n')
        f.write(r'\centering '+'\n')
        f.write(r'\caption{The caption} '+'\n')
        f.write(r'\label{tab:Rlim} '+'\n')
        f.write(r'\begin{tabular}{lcccccccccc} '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\rule{0pt}{2ex} '+'\n')
        f.write(r' & $\ell$ range & \mc{LR16}& \mc{LR24}& \mc{LR33}& \mc{LR42}& \mc{LR53}& \mc{LR63N}& \mc{LR63}& \mc{LR63S}& \mc{LR72} \\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        

        spec = 2 # BB
        speclab = 'BB'

        be = (self.c.cmcbk[LR[0]].be).astype(int)
        bins = ['{:d}--{:d}'.format(be[k],be[k+1]) for k in range(be.size-1)]

        ##########
        # PTE
        f.write(r'\multicolumn{11}{c}{PTE} \\')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')

        for j, binval in enumerate(bins):
            f.write(r' &{0} '.format(binval))

            for k, val in enumerate(LR):

                x = 100*self.c.cmcbk[val].PTE[2,j]                

                if np.isfinite(x):
                    f.write(r'& {:0.1f}'.format(x))
                else:
                    f.write(r'&\ldots')

            f.write(r'\\ '+'\n')
            f.write(r'\addlinespace[1ex] '+'\n')

        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')


        ##########
        # M.L. R_BB
        f.write(r'\multicolumn{11}{c}{Maximum Likelihood $(\rl^{BB})$ } \\')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')

        for j, binval in enumerate(bins):
            f.write(r' &{0} '.format(binval))

            for k, val in enumerate(LR):

                R = self.c.cmcbk[val].R[2,j]                

                up68 = self.c.cmcbk[val].up68[2,j] - R
                up95 = self.c.cmcbk[val].up95[2,j] - R
                down68 = R - self.c.cmcbk[val].down68[2,j] 
                down95 = R - self.c.cmcbk[val].down95[2,j] 

                #x = '${:0.2f}^{{ +{:0.2f}/{:0.2f} }}_{{ -{:0.2f}/{:0.2f} }}$'.format(R,up68,up95,down68,down95)
                x = '${:0.2f}\substack{{ +{:0.2f}/{:0.2f} \\\\ -{:0.2f}/{:0.2f} }}$'.format(R,up68,up95,down68,down95)
                if ~np.isfinite(R):
                    x = r'\ldots'

                f.write(r'& {:s}'.format(x))

            f.write(r'\\ '+'\n')
            f.write(r'\addlinespace[1ex] '+'\n')

        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\toprule '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')

        ############
        # 95% L.L
        f.write(r'\multicolumn{11}{c}{95$\%$~(\textit{99.7}$\%$) Lower Limit $(\rl^{BB})$ } \\')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')

        for j, binval in enumerate(bins):
            f.write(r' &{0} '.format(binval))

            for k, val in enumerate(LR):

                R = self.c.cmcbk[val].R[2,j]                

                like = self.c.cmcbk[val].Rlike[:,2,j]
                Rtrial = self.c.cmcbk[val].Rtrial
                
                LL95 = np.interp(0.05, like, Rtrial)
                dobold = False

                if LL95>=1:
                    LL95 = np.interp(1-0.997, like, Rtrial)
                    dobold = True

                if np.isfinite(R):
                    if dobold:
                        x = '($>$\\textit{{ {:0.3f} }})'.format(LL95)
                    else:
                        x = '$>{:0.3f}$'.format(LL95)
                else:
                    x = r'\ldots'

                f.write(r'& {:s}'.format(x))


            f.write(r'\\ '+'\n')
            f.write(r'\addlinespace[0.5ex] '+'\n')

        f.write(r'\toprule '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')


        f.write(r'\end{tabular} '+'\n')
        f.write(r'\end{table*} '+'\n')

        f.close()

