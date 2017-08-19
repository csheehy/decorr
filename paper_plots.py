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

    def __init__(self):
        # Load data

        self.cqu = {}
        self.cmc = {}
        self.cmc2 = {}
        self.cmc2f = {}

        self.cmclin = {}

        self.cmcbk = {}

        LR = decorr.getLR()

        for k,val in enumerate(LR):

            print(val)

            dir = 'spec/gaussian'

            fnqu = '{0}/synch_therm_{1}_qucov_noise_xpol_xxxx.pickle'.format(dir,val)
            fnmc = '{0}/synch_therm_{1}_mcplusexcess_noise_xpol_xxxx.pickle'.format(dir,val)
            fndb = '{0}/synch_therm_LR72_mcplusexcess_noise_xpol_xxxx.pickle'.format(dir)

            self.cqu[val] = decorr.Calc(fnqu, bintype='planck', full=False, dodebias=False)
            self.cmc[val] = decorr.Calc(fnqu, bintype='planck', full=False, dodebias=fndb)
            self.cmc2[val] = decorr.Calc(fnmc, bintype='planck', full=False, dodebias=fndb)
            self.cmc2f[val] = decorr.Calc(fnmc, bintype='planck', full=True, dodebias=fndb)


            self.cmclin[val] = decorr.Calc(fnmc, bintype='lin', full=True, lmin=0, lmax=700, nbin=70, dodebias=fndb)
            self.cmcbk[val] = decorr.Calc(fnmc, bintype='bk', full=True, dodebias=fndb)


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
        self.plotnh()
        self.plotrawspec()
        self.plotnoispec()
        self.PTEtable()
        self.Rlimittable()
        self.MCPTE()
        self.plotRfine()
        self.plotfulldiff()


    def plotspec(self, LR='LR63'):

        cqu = self.c.cqu[LR]
        cmc = self.c.cmc[LR]
        cmclin = self.c.cmclin[LR]

        # Get model
        mod = decorr.Model(fsky=cqu.spec.fsky)
        modbin = decorr.Model(fsky=cqu.spec.fsky, be=cqu.be)

        spec = ['TT', 'EE','BB']

        close(1)
        figure(1, figsize=(5,8))

        for k in [1,2]:

            subplot(2,1,k)

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
            errorbar(cmc.bc, cmc.R[k], yerr=cmc.err[k], xerr=xerr, fmt='o',
                     label='data after debias', linewidth=2, color='k', ms=4.0, capthick=1, capsize=3) 


            plot([cqu.be[0],cqu.be[-1]],[1,1],':k')

            xlim(cqu.be[0]-3, cqu.be[-1]+3)
            if k==1:
                ylim(0.6,1)
                yticks(np.arange(0.7,1.01,0.1))
            else:
                ylim(0.7,1.15)
                yticks(np.arange(0.7,1.15,0.1))
            
            if k==1:
                legend(loc='upper right')
                

            xlabel(r'Multipole $\ell$')
            if k==2:
                ylabel(r'$\mathcal{R}_{\ell}^{BB}$')
            else:
                ylabel(r'$\mathcal{R}_{\ell}^{EE}$')
                
            ax = gca()
                        
            
            if k==1:
                setp(ax.get_xticklabels(), visible=False)

        subplots_adjust(hspace=0, wspace=0)
        savefig('figures/{0}spec.pdf'.format(LR))

        

    def plotnh(self, bin=0, type='pipl'):

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
            if type == 'pipl':
                Rqu.append(self.c.cqu[val].R[2,bin])
                Rmc.append(getattr(self.c,'cmc')[val].R[2,bin])
                SN.append(getattr(self.c,'cqu')[val].SN[:,2,bin])
                err.append(getattr(self.c,'cqu')[val].err[2,bin])
                be = getattr(self.c,'cmc')[val].be[bin:(bin+2)].astype(int)

            if type == 'bk':
                if (bin == 0) & (val in ['LR16','LR24','LR33']):
                    Rmc.append(np.nan)
                else:
                    Rmc.append(getattr(self.c,'cmcbk')[val].R[2,bin])
                SN.append(getattr(self.c,'cmcbk')[val].SN[:,2,bin])
                err.append(getattr(self.c,'cmcbk')[val].err[2,bin])
                be = getattr(self.c,'cmcbk')[val].be[bin:(bin+2)].astype(int)                    

        if type=='pipl':
            Rqu = np.array(Rqu)
        Rmc = np.array(Rmc)
        SN = np.array(SN).T
        err = np.array(err)


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
        nplots=20
        colormap = cm.nipy_spectral 
        ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1, nplots)])
        h1 = plot(nh, SN[0:nplots].T, alpha=0.3, color='m')

        # Plot real
        hmc = plot(nh, Rmc, 'Dr', markersize=4.0, markeredgewidth=1.0)
        if type == 'pipl':
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

        if type == 'pipl':
            handles.append(hqu[0])
            labels.append(u'data')

        handles.append(hmc[0])
        labels.append(u'data after debias')

        legend(handles, labels, loc='upper right')
        tight_layout()


        figure(1)
        savefig('figures/nh_bin{:s}_{:s}_curve.pdf'.format(binlab, type))


        ############################
        ############################
        # Get some statistics
        err = (np.array(up68) - np.array(down68))/2

        chiRmc = np.sum((Rmc-1)/err)
        chi2Rmc = np.sum(((Rmc-1)/err)**2)
        if type=='pipl':
            chiRqu = np.sum((Rqu-1)/err)
            chi2Rqu = np.sum(((Rqu-1)/err)**2)

        chiSN = np.sum((SN-1)/err, axis=1)
        chi2SN = np.sum(((SN-1)/err)**2, axis=1)

        nrlz = chi2SN.size*1.0
        if type=='pipl':
            PTEchi2qu = np.size(np.where(chi2SN < chi2Rqu)[0]) / nrlz
        PTEchi2mc = np.size(np.where(chi2SN < chi2Rmc)[0]) / nrlz
        print('chi2 PTE = {0}'.format(PTEchi2qu))
        if type=='pipl':
            print('chi2 PTE after debias = {0}'.format(PTEchi2mc))

        if type=='pipl':
            PTEchiqu = np.size(np.where(chiSN < chiRqu)[0]) / nrlz
        PTEchimc = np.size(np.where(chiSN < chiRmc)[0]) / nrlz
        print('chi PTE = {0}'.format(PTEchiqu))
        if type=='pipl':
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
        figure(2, figsize=(5,8))
        clf()

        subplot(3,1,1)

        rng = (0, np.max(np.hstack((chi2SN, chi2SN))))
        hist(chi2SN, range=rng, bins=25, label='sims', color=[0.8,0.8,0.8])
        yl=ylim()
        if type=='pipl':
            plot( [chi2Rqu,chi2Rqu], [0,yl[1]], color=[0.7,0.7,0.7], linewidth=3, label='observed')
        plot( [chi2Rmc,chi2Rmc], [0,yl[1]], 'k', linewidth=3, label='observed (after debias)')
        ylim(0.1,yl[1])
        xlabel(r'$\chi^2$')
        ax=gca()
        ax.xaxis.major.locator.set_params(nbins=4) 

        if type=='pipl':
            text(0.95, 0.97, 'PTE = {:0.2f}'.format(PTEchi2qu),
                 horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes)
        text(0.95, 0.9, 'PTE after debias = {:0.2f}'.format(PTEchi2mc),
             horizontalalignment='right', verticalalignment='top', 
             transform=ax.transAxes)

        legend(loc=(0.45,0.47))


        subplot(3,1,2)
        rng = ( np.min(np.hstack((chiSN, chiSN))), np.max(np.hstack((chiSN, chiSN))) )
        hist(chiSN, range=rng, bins=15, label='sims', color=[0.8,0.8,0.8])
        yl=ylim()
        if type == 'pipl':
            plot( [chiRqu,chiRqu], [0,yl[1]], color=[0.7,0.7,0.7], linewidth=3)
        plot( [chiRmc,chiRmc], [0,yl[1]], color=[0,0,0], linewidth=3)
        ylim(0.1,yl[1])
        xlabel(r'$\chi$')

        tight_layout()

        ax=gca()
        ax.xaxis.major.locator.set_params(nbins=4)

        if type=='pipl':
            text(0.95, 0.97, 'PTE = {:0.2f}'.format(PTEchiqu),
                 horizontalalignment='right', verticalalignment='top',
                 transform=ax.transAxes)

        text(0.95, 0.9, 'PTE after debias = {:0.2f}'.format(PTEchimc),
             horizontalalignment='right', verticalalignment='top', 
             transform=ax.transAxes)


        ####################
        # Plot histogram of zero crossings
        subplot(3,1,3)

        N, be, dum = hist(ncross, range=(-0.5,8.5), bins=9, color=[0.8,0.8,0.8])
        yl=ylim()
        plot( [ncrossr,ncrossr], [0,yl[1]], 'k', linewidth=3)
        xlim(-0.5,8.5)
        xlabel(r'$\mathcal{R^{\mathrm{BB}}}$, # of zero crossings') 

        # Binomial distribution
        N = np.arange(9)
        bn = binom(8, 0.5)
        Nexp = bn.pmf(N)*nrlz
        for k,val in enumerate(Nexp):
            if k==0:
                lab = 'uncorrelated\n expectation'
            else:
                lab = None
            plot([be[k],be[k+1]], [val,val], ':', color=[0,0,0], linewidth=1, label=lab)
        legend()
        yl = ylim()
        ylim(0.2,yl[1])

        ##################
        # Save figs

        tight_layout()
        figure(2)
        savefig('figures/nh_bin{:s}_dist.pdf'.format(binlab))


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

        LR = decorr.getLR()

        close(1)
        fig = figure(1, figsize=(10,8))

        for k,val in enumerate(LR):
            subplot(3,3,k+1)
            ax = gca()

            l = self.c.cmclin[val].bc
            snmc = self.c.cmclin[val].sn[:,:,2,:].mean(0)
            yerr = self.c.cmclin[val].sn[:,:,2,:].std(0)
            r = self.c.cmclin[val].r[:,2,:]

            fac = 1e12
            errorbar(l, r[0]*fac, yerr[0]*fac, fmt='.b')
            ax.set_xscale("log", nonposx='clip')
            ax.set_yscale("log", nonposy='clip')
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

        nrlz =  self.c.cmclin[LR].nrlz
        l = self.c.cmclin[LR].bc
        fac = 1

        nmc = fac * self.c.cmclin[LR].n.mean(0)*1e12
        nmcerr = fac * self.c.cmclin[LR].n.std(0) / np.sqrt(nrlz) * 1e12

        close(1)
        f = figure(1, figsize=(5,9))
        xt = [100,500]

        sp = 2 # 1/2 -> EE/BB

        # c = 0,1,2 -> <217x217> / <353x353> / <217x353>
        for c in [0,1,2]:

            if c==0:
                tit = r'$<217 \mathrm{x} 217>$'
            if c==1:
                tit = r'$<353 \mathrm{x} 353>$'
            if c==2:
                tit = r'$<217 \mathrm{x} 353>$'

            ymc = nmc[c,sp,:]
            dymc = nmcerr[c,sp,:]

            subplot(3,1,c+1)

            semilogx(l, ymc, color='gray')
            fill_between(l, ymc-dymc, ymc+dymc, color=[0.6,0.6,0.6], alpha=0.5)
            plot(l, ymc, 'k')

            semilogx([50,700],[0,0],'k:')


            xlim(50,700)
            ax = gca()
            ax.set_xticks(xt)
            ax.set_xticklabels(xt)

            if c<2:
                setp(ax.get_xticklabels(), visible=False)
            if c != 1:
                ylim(-0.3,0.3)
                ax.set_yticks([-0.2,-0.1,0,0.1,0.2])
            else:
                ylim(-3,3)
                ax.set_yticks([-2,-1,0,1,2])

            ax = gca()


            if c<2:
                setp(ax.get_xticklabels(), visible=False)

            text(0.03, 0.95, tit, color='k',
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax.transAxes)


        subplots_adjust(hspace=0, wspace=0)

        ax = f.add_axes( [0., 0., 1, 1] )
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(.04, 0.5, r'$\ell(\ell+1)\mathcal{C}_{\ell,\mathrm{Noise}}^{BB}/2\pi \ [\mu \mathrm{K}^2]$',
                 rotation='vertical', horizontalalignment='center', verticalalignment='center')
        ax.text(.5, 0.05, r'Multipole $\ell$',
                 horizontalalignment='center', verticalalignment='center')

 
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
               

       PTEthresh = 0.05
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

       imshow(corr, interpolation='nearest')
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


    def plotRfine(self, val='LR72'):
        """Plot R in fine bins"""
        
        ####
        close(1)
        figure(1, figsize=(7,9) )

        bclin = self.c.cmclin[val].bc
        bcpipl = self.c.cmc2f[val].bc
        bcbk = self.c.cmcbk[val].bc

        belin = self.c.cmclin[val].be
        bepipl = self.c.cmc2f[val].be
        bebk = self.c.cmcbk[val].be

        Rlin = self.c.cmclin[val].R
        Rpipl = self.c.cmc2f[val].R
        Rbk = self.c.cmcbk[val].R

        errlin = self.c.cmclin[val].err
        errpipl = self.c.cmc2f[val].err
        errbk = self.c.cmcbk[val].err

        
        for k in range(2):
        #for k in range(1): # Don't plot inset

            if k==1:
                fig = gcf()
                ax = gca()
                pos = ax.get_position()
                fig.add_axes([pos.x0+.08, pos.y0+.04, 0.25, 0.3])

            xerrlin = (belin[1:] - belin[0:-1])/2
            xerrpipl = (bepipl[1:] - bepipl[0:-1])/2
            xerrbk = (bebk[1:] - bebk[0:-1])/2

            if k==0:
                ylim(0.6,1.3)
                xlim(19,336)
                #xlim(19,700)

            else:
                ylim(0.98,1.02)
                xlim(19,125)
                ax = gca()
                ax.set_yticks([0.98,1,1.02])

            errorbar(bclin, Rlin[2], errlin[2], xerrlin, fmt='.',c=[0.7,0.7,0.7], label=r'$\Delta\ell = 10$')
            errorbar(bcbk, Rbk[2], errbk[2], xerrbk, fmt='ok', ms=4, label=r'$\Delta\ell = 35$', lw=2, capsize=4)
            errorbar(bcpipl-1, Rpipl[2], errpipl[2], xerrpipl, fmt='s', c='r', ms=4, label='PIP-L binning')

            plot([0,350],[1,1],'k:')

            if k==0:
                legend(loc='upper left')
                xlabel(r'Multipole $\ell$')
                ylabel(r'$\mathcal{R}_{BB}$')

    
        savefig('figures/Rfine_{:s}.pdf'.format(val))

        #######
        for mm in range(2):

            close(mm+2)
            fig = figure(mm+2, figsize=(7,10))

            if mm==0:
                yl = [ [-12,6], [-12,6], [-12,6], [-12,6], [-2,6], [-2,6], [-2,6], [-2,6], [-2,6] ]
                xl = [20,700]
            else:
                yl = [ [-2,2], [-2,2], [-2,2], [-2,2], [0.5,1.5], [0.5,1.5], [0.5,1.5], [0.5,1.5], [0.5,1.5]]
                xl = [20,320]

            LR = decorr.getLR()

            for k,val in enumerate(LR):

                Rlin = self.c.cmclin[val].R
                errlin = self.c.cmclin[val].err

                subplot(5,2,k+1)

                # Plot planck bin
                for m,val2 in enumerate(bepipl):
                    plot([val2,val2],[yl[k][0],yl[k][1]],'--', color='gray')
                plot([0,700],[1,1],':k')

                errorbar(bclin, Rlin[2], errlin[2], fmt='.k', label=val)
                xlim(*xl)

                yll = np.array(yl[k])
                yll[0] = yll[0]+1e-3
                yll[1] = yll[1]-1e-3
                ylim(yl[k])

                ax = gca()
                yt = ax.get_yticks()
                ax.set_yticks(yt[1:-1])

                if (k+1)<9:
                    setp(ax.get_xticklabels(), visible=False)
                if np.mod(k,2) == 1:
                    setp(ax.get_yticklabels(), visible=False)

                text(0.03, 0.95, val, color='k',
                     horizontalalignment='left', verticalalignment='top',
                     transform=ax.transAxes)


            subplots_adjust(hspace=0, wspace=0)

            ax = fig.add_axes( [0., 0., 1, 1] )
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(.04, 0.5, r'$\mathcal{R_{BB}}$',
                     rotation='vertical', horizontalalignment='center', verticalalignment='center')
            ax.text(.5, 0.05, r'Multipole $\ell$',
                     horizontalalignment='center', verticalalignment='center')

            figure(mm+2)
            savefig('figures/Rfine_allLR_{:d}.pdf'.format(mm))








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

        be1 = (self.c.cmcbk[LR[0]].be).astype(int)
        bins1 = ['{:d}--{:d}'.format(be1[k],be1[k+1]) for k in range(be1.size-1)]

        be2 = (self.c.cmc2f[LR[0]].be[2:]).astype(int)
        bins2 = ['{:d}--{:d}'.format(be2[k],be2[k+1]) for k in range(be2.size-1)]

        bins = np.hstack( (bins1,bins2) )

        ##########
        # PTE
        f.write(r'\multicolumn{11}{c}{PTE} \\')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\toprule '+'\n')

        for j, binval in enumerate(bins):
            f.write(r' &{0} '.format(binval))

            for k, val in enumerate(LR):
                
                if j<9:
                    x = 100*self.c.cmcbk[val].PTE[2,j]
                else:
                    x = 100*self.c.cmc2f[val].PTE[2,j-7]

                xx = r'& {:0.1f}'.format(x)

                if ~np.isfinite(x):
                    xx = r'&\ldots'
                
                if (j==0) & (val in ['LR16','LR24', 'LR33']):
                    xx = r'&\ldots'

                f.write(xx)

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

                if j<9:
                    R = self.c.cmcbk[val].R[2,j]
                    up68 = self.c.cmcbk[val].up68[2,j] - R
                    up95 = self.c.cmcbk[val].up95[2,j] - R
                    down68 = R - self.c.cmcbk[val].down68[2,j]
                    down95 = R - self.c.cmcbk[val].down95[2,j]
                    be = self.c.cmcbk[val].be[j:(j+2)]
                else:
                    R = self.c.cmc2f[val].R[2,j-7]
                    up68 = self.c.cmc2f[val].up68[2,j-7] - R
                    up95 = self.c.cmc2f[val].up95[2,j-7] - R
                    down68 = R - self.c.cmc2f[val].down68[2,j-7] 
                    down95 = R - self.c.cmc2f[val].down95[2,j-7]
                    be = self.c.cmc2f[val].be[(j-7):(j-7+2)]

                # Transform everything to dust only R
                fsky = self.c.cmcbk[val].spec.fsky
                mod = decorr.Model(fsky=fsky, be=be)
                up68, dum = mod.getR(Robs=up68)
                up95, dum = mod.getR(Robs=up95)
                down68, dum = mod.getR(Robs=down68)
                down95, dum = mod.getR(Robs=down95)
                up68 = up68[0]; 
                up95 = up95[0]; 
                down68 = down68[0]; 
                down95 = down95[0]; 
                    
                x = '${:0.3f}\substack{{ +{:0.2f}\\\\ -{:0.2f} }}$'.format(R,up95/2,down95/2)

                if ((up95/2)<.1) & ((down95/2)<.1):
                    err1 = ('{:.3f}'.format(up68/2)).replace('0.','.')
                    err2 = ('{:.3f}'.format(up95/2)).replace('0.','.')
                    err3 = ('{:.3f}'.format(down68/2)).replace('0.','.')
                    err4 = ('{:.3f}'.format(down95/2)).replace('0.','.')
                    x = '${:0.3f}\substack{{ +{:s} \\\\ -{:s} }}$'.format(R,err2,err4)


                if ~np.isfinite(R):
                    x = r'\ldots'
                if (j==0) & (val in ['LR16','LR24', 'LR33']):
                    x= r'\ldots'
                if ((R + up95) < 1) | ((R - down95)>1):
                    x = r'\boldmath ' + x 

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

                if j<9:
                    R = self.c.cmcbk[val].R[2,j]                
                    like = self.c.cmcbk[val].Rlike[:,2,j]
                    Rtrial = self.c.cmcbk[val].Rtrial
                    be = self.c.cmcbk[val].be[j:(j+2)]
                else:
                    R = self.c.cmc2f[val].R[2,j-7]                
                    like = self.c.cmc2f[val].Rlike[:,2,j-7]
                    Rtrial = self.c.cmc2f[val].Rtrial
                    be = self.c.cmc2f[val].be[(j-7):(j-7+2)]

                LL95 = np.interp(0.05, like, Rtrial)
                dobold = False


                if LL95>=1:
                    LL95 = np.interp(1-0.997, like, Rtrial)
                    dobold = True

                # Transform lower limit to dust only R
                fsky = self.c.cmcbk[val].spec.fsky
                mod = decorr.Model(fsky=fsky, be=be)
                LL95, dum = mod.getR(Robs=LL95)
                LL95 = LL95[0]

                if dobold:
                    x = '($>$\\textit{{ {:0.3f} }})'.format(LL95)
                else:
                    x = '$>{:0.3f}$'.format(LL95)

                if ~np.isfinite(R):
                    x = r'\ldots'
                if (j==0) & (val in ['LR16','LR24', 'LR33']):
                    x= r'\ldots'

                f.write(r'& {:s}'.format(x))


            f.write(r'\\ '+'\n')
            f.write(r'\addlinespace[0.5ex] '+'\n')

        f.write(r'\toprule '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')


        f.write(r'\end{tabular} '+'\n')
        f.write(r'\end{table*} '+'\n')

        f.close()


    def plotfulldiff(self):
        """Plot difference of mc full and mc non-full real and sim"""

        LR = decorr.getLR()

        cf = self.c.cmc2f
        c = self.c.cmc2
        bc = self.c.cmc2[LR[0]].bc

        clf()
        
        yl = [ [-2,2], [-2,2], [-0.5,0.5], [-0.5,0.5], [-0.2,0.2], [-0.2,0.2], [-0.1,0.1], [-0.1,0.1], [-0.1,0.1] ]

        close(1)
        fig = figure(1, figsize=(6,8))

        for k,val in enumerate(LR):

            subplot(5,2,k+1)

            # Difference 
            diffr = cf[val].R - c[val].R
            diffsn = cf[val].SN - c[val].SN
            

            # Percentiles
            down99p7 = np.nanpercentile(diffsn, 50 - 99.7/2, 0)
            down95 = np.nanpercentile(diffsn, 50 - 95./2, 0)
            down68 = np.nanpercentile(diffsn, 50 - 68./2, 0)
            up99p7 = np.nanpercentile(diffsn, 50 + 99.7/2, 0)
            up95 = np.nanpercentile(diffsn, 50 + 95./2, 0)
            up68 = np.nanpercentile(diffsn, 50 + 68./2, 0)

            fill_between(bc, down99p7[2], up99p7[2], color=[0.8,0.8,0.8])
            fill_between(bc, down95[2], up95[2], color=[0.5,0.5,0.5])
            fill_between(bc, down68[2], up68[2], color=[0.3,0.3,0.3])

            plot(bc, diffr[2],'k.')

            xlim(50,700)
            ylim(*yl[k])

            ax = gca()
            
            rng = yl[k][1] - yl[k][0]
            ax.set_yticks([-rng/4, 0, +rng/4])
            
            
            if (k+1)<9:
                setp(ax.get_xticklabels(), visible=False)
            if np.mod(k,2) == 1:
                setp(ax.get_yticklabels(), visible=False)

            text(0.03, 0.95, val, color='k',
                 horizontalalignment='left', verticalalignment='top',
                 transform=ax.transAxes)
                     
        subplots_adjust(hspace=0, wspace=0)
        
        ax = fig.add_axes( [0., 0., 1, 1] )
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(.02, 0.5, r'$\Delta \mathcal{R_{BB}}$',
                 rotation='vertical', horizontalalignment='center', verticalalignment='center')
        ax.text(.5, 0.05, r'Multipole $\ell$',
                 horizontalalignment='center', verticalalignment='center')

        savefig('figures/fulldiff.pdf')
