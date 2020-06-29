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

        self.cqu   = {}
        self.cmc   = {}
        self.cquds = {}
        self.cmcds = {}

        self.cmcnoi = {}

        self.cmclin   = {}
        self.cmclinds = {}


        self.cmcbk   = {}
        self.cqubk   = {}
        self.cmcbkds = {}
        self.cqubkds = {}

        self.cd0 = {}
        self.cd1 = {}
        self.cd2 = {}
        
        self.cd0pl = {}
        self.cd0bk = {}

        self.chmds1 = {}
        self.chmds2 = {}

        self.chmds1bk = {}
        self.chmds2bk = {}

        self.chmds1dl3 = {}
        self.chmds2dl3 = {}

        LR = decorr.getLR()

        for k,val in enumerate(LR):

            print(val)

            dir = 'spec/gaussian'
            dird0 = 'spec/dust0_tophat'
            dird1 = 'spec/dust1_tophat'
            dird2 = 'spec/dust2_tophat'


            fnqu   = '{0}/synch_therm_{1}_qucov_noise_xpol_xxxx.npz'.format(dir,val)
            fnquds = '{0}/synch_therm_{1}_qucovds_noise_xpol_xxxx.npz'.format(dir,val)

            fnmc   = '{0}/synch_therm_{1}_mcplusexcess_noise_xpol_xxxx.npz'.format(dir,val)
            fndb   = '{0}/synch_therm_{1}_mcplusexcess_noise_xpol_xxxx.npz'.format(dir,val)

            fnd0   = '{0}/synch_therm_{1}_mcplusexcess_noise_xxxx.pickle'.format(dird0,val)
            fnd1   = '{0}/synch_therm_{1}_mcplusexcess_noise_xxxx.pickle'.format(dird1,val)
            fnd2   = '{0}/synch_therm_{1}_mcplusexcess_noise_xxxx.pickle'.format(dird2,val)

            fnhmds1 = 'spec/dust0_tophat/synch_therm_{:s}_noihmds1_xxxx.pickle'.format(val)
            fnhmds2 = 'spec/dust0_tophat/synch_therm_{:s}_noihmds2_xxxx.pickle'.format(val)

            self.cqu[val]    = decorr.Calc(fnqu,   bintype='planck', dodebias=False)
            self.cmc[val]    = decorr.Calc(fnqu,   bintype='planck', dodebias=fndb)
            self.cquds[val]  = decorr.Calc(fnquds, bintype='planck', dodebias=False)
            self.cmcds[val]  = decorr.Calc(fnquds, bintype='planck', dodebias=fndb)

            self.cmcnoi[val] = decorr.Calc(fnmc, bintype='lin', lmin=0, lmax=700, nbin=70, dodebias=False)

            self.cmclin[val]   = decorr.Calc(fnqu,   bintype='lin', lmin=0, lmax=700, nbin=70, dodebias=fndb)
            self.cmclinds[val] = decorr.Calc(fnquds, bintype='lin', lmin=0, lmax=700, nbin=70, dodebias=fndb)

            self.cqubk[val]   = decorr.Calc(fnqu,   bintype='bk', dodebias=False)
            self.cmcbk[val]   = decorr.Calc(fnqu,   bintype='bk', dodebias=fndb)
            self.cqubkds[val] = decorr.Calc(fnquds, bintype='bk', dodebias=False)
            self.cmcbkds[val] = decorr.Calc(fnquds, bintype='bk', dodebias=fndb)

            self.cd0[val] = decorr.Calc(fnd0, bintype='lin', lmin=0, lmax=700, nbin=70, dodebias=False)
            self.cd1[val] = decorr.Calc(fnd1, bintype='lin', lmin=0, lmax=700, nbin=70, dodebias=False)
            self.cd2[val] = decorr.Calc(fnd2, bintype='lin', lmin=0, lmax=700, nbin=70, dodebias=False)

            self.cd0pl[val] = decorr.Calc(fnd0, bintype='planck', dodebias=False)
            self.cd0bk[val] = decorr.Calc(fnd0, bintype='bk', dodebias=False)

            if val=='LR72':
                self.chmds1[val] = decorr.Calc(fnhmds1, bintype='planck', dodebias=False)
                self.chmds2[val] = decorr.Calc(fnhmds2, bintype='planck', dodebias=False)
                self.chmds1bk[val] = decorr.Calc(fnhmds1, bintype='bk', dodebias=False)
                self.chmds2bk[val] = decorr.Calc(fnhmds2, bintype='bk', dodebias=False)



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

        self.plotspec()
        self.plotnh()
        self.plotnhzerocross()
        self.plotrawspec()
        self.plotnoispec()
        self.Rtable()
        self.plotcorrmatrix()
        self.plotRfine()


    def plotspec(self, LR='LR63'):

        cqu = self.c.cqu[LR]
        cmc = self.c.cmc[LR]

        cquds = self.c.cquds[LR]
        cmcds = self.c.cmcds[LR]

        # Get model
        mod = decorr.Model(fsky=cqu.spec.fsky)
        modbin = decorr.Model(fsky=cqu.spec.fsky, be=cqu.be)

        spec = ['TT', 'EE','BB']

        close(1)
        figure(1, figsize=(5,8))

        for k in [1,2]:

            subplot(2,1,k)

            if k==1:
                y = mod.RE
                ybin = modbin.RE
            if k==2:
                y = mod.RB
                ybin = modbin.RB
                x = np.loadtxt('PIPL_pdf2draw.csv',delimiter=',')

            # Plot bins
            xerr = (cmc.be[1:] - cmc.be[0:-1])/2
            for q in range(cmc.nbin):
                plot([cmc.be[q],cmc.be[q+1]],[ybin[q],ybin[q]],color='gray')

            # Plot model
            plot(mod.l, y, 'k--', label='model')

            # Plot binned model
            plot(cqu.bc, ybin,'.', color='gray', label='expectation', ms=4)


            # Plot real data
            plot(cqu.bc-16, cqu.R[k], '.r', label='HM (no debias)')
            errorbar(cmc.bc-8, cmc.R[k], yerr=cmc.err[k], fmt='D',
                     label='HM', linewidth=1, color='r', ms=4.0, capthick=1, capsize=3) 

            plot(cquds.bc+8, cquds.R[k], '.b', label='DS (no debias)')
            errorbar(cmcds.bc+16, cmcds.R[k], yerr=cmcds.err[k], fmt='s',
                     label='DS', linewidth=1, color='b', ms=4.0, capthick=1, capsize=3) 


            plot([cqu.be[0],cqu.be[-1]],[1,1],':k')

            ## Plot Planck data
            if k==2:
                LRR = np.array(['LR16','LR24','LR33','LR42','LR53','LR63N','LR63','LR63S','LR72'])
                indd = np.where(LRR==LR)[0][0]
                h = plot(cqu.bc-16, x[indd,:], 'xr', label='HM (PIPL)',)
                legend([h[0]],['HM (PIPL)'])

            xlim(cqu.be[0]-3, cqu.be[-1]+3)
            if k==1:
                ylim(0.6,1)
                yticks(np.arange(0.6,1.01,0.1))
            else:
                ylim(0.6,1.2)
                yticks(np.arange(0.6,1.11,0.1))
            
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
        savefig('figures/{0}spec.pdf'.format(LR), bbox_inches='tight')

        

    def plotnh(self):

        f = open('specsyst.pickle','rb')
        hmds = cP.load(f)
        f.close()

        bins = [0,1,2,3,4,0,1]
        yll = [ [], [.5,1.5], [.5,1.5], [0,3.0], [0,3.0], [.7,1.5],[0,3.0]]
        types = ['bk', 'bk','bk','bk','bk','pipl','pipl']
        pos = [0,1,2,3,4,5,6]
        npan = len(bins)

        Rsystmean = []

        #################
        close(1)
        figure(1, figsize=(10,8))
        clf()


        for kk in range(npan):

            handles = []
            labels = []


            for jj in range(2):

                # Bar width
                wd = 0.1

                if jj==0:
                    # HM
                    dw = -wd/2
                    ds = ''
                    c = 'r'
                    sym = 'D'
                    dh = 0.08
                if jj==1:
                    # DS
                    dw = +wd/2
                    ds = 'ds'
                    c = 'b'
                    sym = 's'
                    dh = 0

                bin = bins[kk]
                type = types[kk]

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
                Rsystlo = []
                Rsysthi = []

                mod = decorr.Model()



                for k,val in enumerate(LR):

                    if type == 'pipl':
                        Rqu.append(getattr(self.c,'cqu'+ds)[val].R[2,bin])
                        Rmc.append(getattr(self.c,'cmc'+ds)[val].R[2,bin])
                        SN.append(getattr(self.c,'cqu'+ds)[val].SN[:,2,bin])
                        err.append(getattr(self.c,'cqu'+ds)[val].err[2,bin])
                        be = getattr(self.c,'cmc'+ds)[val].be[bin:(bin+2)].astype(int)

                    if type == 'bk':
                        Rqu.append(getattr(self.c,'cqubk'+ds)[val].R[2,bin])
                        Rmc.append(getattr(self.c,'cmcbk'+ds)[val].R[2,bin])
                        SN.append(getattr(self.c,'cqubk'+ds)[val].SN[:,2,bin])
                        err.append(getattr(self.c,'cqubk'+ds)[val].err[2,bin])
                        be = getattr(self.c,'cmcbk'+ds)[val].be[bin:(bin+2)].astype(int)                    

                    if jj==0:

                        dl = be[1]-be[0]
                        if dl == 35:
                            dlhi = 3
                            dllo = 18
                        else:
                            dlhi = 3
                            dllo = dl/2

               
                        syst = np.zeros((3,3,721))
                        drr = np.sqrt(np.nanmean((hmds[0][kk][dlhi]['r'][:,2,:] - hmds[1][kk][dlhi]['r'][:,2,:])**2, 1))
                        dss = np.nanmean(np.sqrt(np.nanmean((hmds[0][kk][dlhi]['sn'][:,:,2,:] - hmds[1][kk][dlhi]['sn'][:,:,2,:])**2, 2)),0)
                        fac = 2*np.pi / (be.mean()*(be.mean()+1))
                        syst[0,2,:] = 1e12*(drr - dss)[0] * fac/2
                        syst[1,2,:] = 1e12*(drr - dss)[1] * fac/2
                        syst[2,2,:] = 1e12*(drr - dss)[2] * fac/2
                        syst[np.where(syst<0)] = 0
                        RB, dum = mod.getR(fsky=decorr.LR2fsky(val), spec=2, be=be, syst=syst)
                        Rsysthi.append(RB)

                        syst = np.zeros((3,3,721))
                        drr = np.sqrt(np.nanmean((hmds[0][kk][dllo]['r'][:,2,:] - hmds[1][kk][dllo]['r'][:,2,:])**2, 1))
                        dss = np.nanmean(np.sqrt(np.nanmean((hmds[0][kk][dllo]['sn'][:,:,2,:] - hmds[1][kk][dllo]['sn'][:,:,2,:])**2, 2)),0)
                        fac = 2*np.pi / (be.mean()*(be.mean()+1))
                        syst[0,2,:] = 1e12*(drr - dss)[0] * fac/2
                        syst[1,2,:] = 1e12*(drr - dss)[1] * fac/2
                        syst[2,2,:] = 1e12*(drr - dss)[2] * fac/2
                        syst[np.where(syst<0)] = 0
                        RB, dum = mod.getR(fsky=decorr.LR2fsky(val), spec=2, be=be, syst=syst)
                        Rsystlo.append(RB)


                Rqu = np.array(Rqu)
                Rmc = np.array(Rmc)
                SN = np.array(SN).T
                err = np.array(err)
                Rsystlo = np.array(Rsystlo)
                Rsysthi = np.array(Rsysthi)


                if jj==0:
                    Rsystmean.append( (Rsysthi + Rsystlo)/2.)
                    
                    
                if pos[kk]==0:
                    continue

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



                err = (np.array(up68) - np.array(down68))/2
                chiR =   np.nansum((Rmc  -1)/err)
                chi2R   = np.nansum(((Rmc  -1)/err)**2)
                ind = np.isfinite(Rmc)
                chiSN   = np.sum((SN[:,ind]  -1)/err[ind], axis=1)
                chi2SN   = np.sum(((SN[:,ind]  -1)/err[ind])**2, axis=1)
                nrlz   = np.where(np.isfinite(chi2SN))[0].size*1.0
                PTEchi2   = np.size(np.where(chi2SN   > chi2R)[0]) / nrlz
                PTEchi   = np.size(np.where(chiSN   < chiR)[0]) / nrlz

                # Begin plotting
                subplot(np.ceil(npan/2.)-1,2,pos[kk])

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

                    bar(left+dw, height95, width=wd, bottom=bottom95, color='0.8',
                        edgecolor='0.4', label=lab95)
                    bar(left+dw, height68, width=wd, bottom=bottom68, color='0.5',
                        edgecolor='0.4', label=lab68)
                    plot([left+dw,left+wd+dw],[med[k], med[k]], color='0.3', linewidth=2, label=labmed)

                ax = gca()

                # Plot a few realizations
                nplots=20
                colormap = cm.nipy_spectral 
                ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1, nplots)])
                if jj==0:
                    h1 = plot(nh+dw, SN[0:nplots].T, alpha=0.3, color=[1,.3,.3])

                # Plot systematics
                if jj==0:
                    #hsys = plot(nh, Rsysthi, '-.k', label='systematics')
                    #hsys = plot(nh, Rsystlo, '-.k')
                    #hsys = plot(nh, 2-Rsysthi, '-.k', label='systematics')
                    #hsys = plot(nh, 2-Rsystlo, '-.k')
                    hsys = fill_between(nh, Rsysthi[:,0],    Rsystlo[:,0],    color='c',alpha=0.3)

                # Plot real
                hmc = plot(nh+dw, Rmc, sym+c, markersize=4.0, markeredgewidth=1.0)
                hqu = plot(nh+dw, Rqu, '.'+c)
                plot([0,7],[1,1],'k:')

                if pos[kk] in [5,6]:
                    xlabel(r'$N_H/10^{20} [cm^{-2}]$')
                else:
                    setp(gca().get_xticklabels(), visible=False)

                ylim(*yll[kk])
                if (pos[kk] in [3,4]) & (jj==1):
                    gca().set_yticks(gca().get_yticks()[1:-1])
                if (pos[kk] in [5,6]) & (jj==1):
                    gca().set_yticks(gca().get_yticks()[0:-1])
                if (pos[kk] in [5]) & (jj==1):
                    gca().set_yticks(gca().get_yticks()[0::2])

                

                ylabel(r'$\mathcal{R}_{' + binlab + r'}^{BB}$')
                xlim(1,7)

                # Plot planck data
                if (type=='pipl') & (ds==''):
                    x = np.loadtxt('PIPL_pdf2draw.csv', delimiter=',')
                    h3 = plot(nh+dw, x[:,bin], 'xr')
                    legend(h3,['HM (PIPL)'])

                # Add PTE text
                text(0.95, 0.02+dh, 
                     r'PTE $(\chi, \chi^2)$ = ({:0.3f}, {:0.3f})'.format(PTEchi,PTEchi2),
                     horizontalalignment='right', verticalalignment='bottom', 
                     transform=ax.transAxes, color=c)

                if jj==0:

                    handles.append(hqu[0])
                    labels.append(u'HM (no debias)')

                    handles.append(hmc[0])
                    labels.append(u'HM')

                    handles.append(h1[0])
                    labels.append(u'HM sims')


                else:

                    handles.append(hqu[0])
                    labels.append(u'DS (no debias)')

                    handles.append(hmc[0])
                    labels.append(u'DS')

                    handles.append(hsys)
                    labels.append(u'systematics')


                    
            if kk==1:
                legend(handles, labels, loc='upper right',ncol=2)


        tight_layout()
        subplots_adjust(hspace=0)

        figure(1)
        savefig('figures/nh_curve.pdf', bbox_inches='tight')

        self.Rsystmean = Rsystmean

    def plotnhzerocross(self):

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

        bin = 1
        type = 'cmcbk'

        close(1)
        figure(1, figsize=(4,4))
        clf()

        # Get data
        R = []
        Rds = []
        SN = []
        SNds = []


        for k,val in enumerate(LR):

            R.append(getattr(self.c,type)[val].R[2,bin])
            Rds.append(getattr(self.c,type+'ds')[val].R[2,bin])
            SN.append(getattr(self.c,type)[val].SN[:,2,bin])
            SNds.append(getattr(self.c,type+'ds')[val].SN[:,2,bin])

        R = np.array(R)
        Rds = np.array(Rds)
        SN = np.array(SN).T
        SNds = np.array(SNds).T

        s = getattr(self.c,type)[val].be[bin]
        e = getattr(self.c,type)[val].be[bin+1]

        ############################
        ############################
        # Get some statistics

        ####################
        # Plot histogram of zero crossings

        # Calculate number of zero crossings
        def nzero(x):
            if np.ndim(x) == 1:
                x = np.reshape(x, [1, x.size])
            sgx = np.sign(x-1)
            cross = (sgx[:,1:] != sgx[:,0:-1]).astype(float)
            ncross = np.sum(cross, axis=1)
            return ncross

        ncross = nzero(SN)
        ncrossr = nzero(R)

        ncrossds = nzero(SNds)
        ncrossrds = nzero(Rds)

        N, be, dum =   hist(ncross,   range=(-0.5,8.5), bins=9, color='r', alpha=0.8, normed=True)
        Nds, be, dum = hist(ncrossds, range=(-0.5,8.5), bins=9, color='b', alpha=0.5, normed=True)
        yl=ylim()
        plot( [ncrossr,ncrossr],     [0,yl[1]], 'r', linewidth=3, label='HM')
        plot( [ncrossrds+0.02,ncrossrds+0.02], [0,yl[1]], 'b', linewidth=3, label='DS')
        xlim(-0.5,8.5)

        # Binomial distribution
        N = np.arange(9)
        bn = binom(8, 0.5)
        Nexp = bn.pmf(N)
        for k,val in enumerate(Nexp):
            if k==0:
                lab = 'uncorr.\n exp.'
            else:
                lab = None
            plot([be[k],be[k+1]], [val,val], ':', color=[0,0,0], linewidth=1, label=lab)
        legend()
        ylim(0,yl[1])

        xlabel(r'$(\mathcal{R}^{BB}_{'+'{:d}-{:d}'.format(s,e) + r'} - 1)$ zero crossings')

        ##################
        # Save figs

        tight_layout()
        figure(1)
        savefig('figures/nh_zerocross.pdf', bbox_inches='tight')


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
                     

        tight_layout()
        subplots_adjust(hspace=0, wspace=0)
        
        ax = fig.add_axes( [0., 0., 1, 1] )
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(.00, 0.5, r'$\ell(\ell+1)\mathcal{C}_{\ell}^{\mathrm{BB}}/2\pi \ [\mu\mathrm{K}^2]$', 
                 rotation='vertical', horizontalalignment='center', verticalalignment='center')
        ax.text(.5, 0.0, r'Multipole $\ell$',
                 horizontalalignment='center', verticalalignment='center')


        savefig('figures/rawspec.pdf', bbox_inches='tight', pad_inches=0)



    def plotnoispec(self, LR='LR63'):

        nrlz =  self.c.cmcnoi[LR].nrlz
        l = self.c.cmcnoi[LR].bc
        fac = 1

        nmc = fac * self.c.cmcnoi[LR].n.mean(0)*1e12
        nmcerr = fac * self.c.cmcnoi[LR].n.std(0) / np.sqrt(nrlz) * 1e12

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

            semilogx([20,700],[0,0],'k:')


            xlim(20,700)
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


        tight_layout()
        subplots_adjust(hspace=0, wspace=0)

        ax = f.add_axes( [0., 0., 1, 1] )
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(.01, 0.5, r'$\ell(\ell+1)\mathcal{C}_{\ell,\mathrm{FFP8}}^{BB}/2\pi \ [\mu \mathrm{K}^2]$',
                 rotation='vertical', horizontalalignment='center', verticalalignment='center')
        ax.text(.5, 0.0, r'Multipole $\ell$',
                 horizontalalignment='center', verticalalignment='center')

 
        savefig('figures/noispec.pdf'.format(c), bbox_inches='tight')


    def plotcorrmatrix(self):
       """Compute PTE table on a realization by realization basis"""
       cmc = dc(self.c.cmc)
       cbk = dc(self.c.cqubk)

       LR = decorr.getLR()

       nLR = len(LR)
       nrlz = cmc[LR[0]].nrlz



       # Plot correlation matrix
       close(1)
       fig = figure(1, figsize=(6,5))

       # Correlation matrix of SN R values
       Rtab = np.zeros((nLR*4, nrlz))
       for j,val in enumerate(LR):
           Rtab[j+np.array([0,9,18,27]), :] = np.hstack( (cbk[val].SN[:,2,1:4],cmc[val].SN[:,2,[0]]) ).T

       # Choose which quantity to get correlation matrix of
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
              ['$55-90$','$90-125$','$125-160$','$50-160$'], rotation=90, multialignment='center')
       tick_params(pad=30, labelsize=10)

       xticks(np.array([9,18,27,36])-5, 
              ['$55-90$','$90-125$','$125-160$','$50-160$'], rotation=0, multialignment='center')
       tick_params(pad=30, labelsize=10)


       savefig('figures/corr_matrix.pdf', bbox_inches='tight')



    def plotRfine(self, val='LR63'):
        """Plot R in fine bins"""
        
        ####
        close(1)
        figure(1, figsize=(7,6) )

        bclin = self.c.cmclin[val].bc
        bcpipl = self.c.cmc[val].bc
        bcbk = self.c.cmcbk[val].bc

        belin = self.c.cmclin[val].be
        bepipl = self.c.cmc[val].be
        bebk = self.c.cmcbk[val].be

        Rlin = self.c.cmclin[val].R
        Rpipl = self.c.cmc[val].R
        Rbk = self.c.cmcbk[val].R

        errlin = self.c.cmclin[val].stderr
        errpipl = self.c.cmc[val].stderr
        errbk = self.c.cmcbk[val].stderr

        Rlinds = self.c.cmclinds[val].R
        Rpiplds = self.c.cmcds[val].R
        Rbkds = self.c.cmcbkds[val].R

        errlinds = self.c.cmclinds[val].stderr
        errpiplds = self.c.cmcds[val].stderr
        errbkds = self.c.cmcbkds[val].stderr

        
        xerrlin = (belin[1:] - belin[0:-1])/2
        xerrpipl = (bepipl[1:] - bepipl[0:-1])/2
        xerrbk = (bebk[1:] - bebk[0:-1])/2

        k=2 # BB

        errorbar(bclin, Rlin[k], errlin[k], fmt='D',c=[.6,.6,.6], label=r'$\Delta\ell = 10$ (HM)', ms=3,markeredgecolor='none')
        errorbar(bcbk-1, Rbk[k], errbk[k], xerrbk, fmt='D', color='r', ms=5, label=r'$\Delta\ell = 35$ (HM)', lw=2, capsize=3,zorder=500)

        errorbar(bclin, Rlinds[k], errlinds[k], fmt='s',c=[.3,.3,.3], label=r'$\Delta\ell = 10$ (DS)', ms=3, markeredgecolor='none')
        errorbar(bcbk+1, Rbkds[k], errbkds[k], xerrbk, fmt='s', color='b', ms=5, label=r'$\Delta\ell = 35$ (DS)', lw=2, capsize=3,zorder=500)

        mod = decorr.Model(fsky=self.c.cqu[val].spec.fsky)

        xlim(19,320)
        plot(self.c.cd0[val].bc,self.c.cd0[val].S.mean(0)[2],':', c='k', label='model (no decorr)')
        plot(self.c.cd1[val].bc,self.c.cd1[val].S.mean(0)[2],'--', c='gray', label='model (PySM d1)')
        plot(self.c.cd2[val].bc,self.c.cd2[val].S.mean(0)[2],'--k', label='model (PySM d2)')
        ylim(0.7,1.3)
        ylabel(r'$\mathcal{R}_{BB}$')
        xlabel(r'Multipole $\ell$')
        legend(loc='upper left')

        savefig('figures/Rfine_{:s}.pdf'.format(val), bbox_inches='tight')

        #######
        for mm in [1]:

            close(mm+2)
            fig = figure(mm+2, figsize=(7,10))

            if mm==0:
                yl = [ [-12,6], [-12,6], [-12,6], [-12,6], [-2,6], [-2,6], [-2,6], [-2,6], [-2,6] ]
                xl = [20,700]
            else:
                yl = [ [-3,4], [-3,4], [-.5,2], [-.5,2], [0,2], [0,2], [0,2], [0,2], [0,2]]
                xl = [20,320]

            LR = decorr.getLR()

            for k,val in enumerate(LR):

                Rlin = self.c.cmclin[val].R
                errlin = self.c.cmclin[val].err
                Rlinds = self.c.cmclinds[val].R
                errlinds = self.c.cmclinds[val].err

                subplot(5,2,k+1)

                # Plot planck bin
                for m,val2 in enumerate(bepipl):
                    plot([val2,val2],[yl[k][0],yl[k][1]],'--', color='gray')
                plot([0,700],[1,1],':k')

                errorbar(bclin, Rlin[2], errlin[2], fmt='Dr', label='HM',ms=3,markeredgecolor='none')
                errorbar(bclin+2, Rlinds[2], errlinds[2], fmt='sb', label='DS',ms=3,markeredgecolor='none')
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

                if k==0:
                    legend()

            tight_layout()
            subplots_adjust(hspace=0, wspace=0)

            ax = fig.add_axes( [0., 0., 1, 1] )
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.text(.0, 0.5, r'$\mathcal{R_{BB}}$',
                     rotation='vertical', horizontalalignment='center', verticalalignment='center')
            ax.text(.5, 0.0, r'Multipole $\ell$',
                     horizontalalignment='center', verticalalignment='center')

            figure(mm+2)
            savefig('figures/Rfine_allLR_{:d}.pdf'.format(mm), bbox_inches='tight')


    def Rtable(self):
        """Plot R ML and PTE"""

        LR = decorr.getLR()

        # Reorder
        LR = np.array(LR)[ [0,1,2,3,4,5,7,6,8] ]

        capML = r"""Noise debiased $\rl^{BB}$ in different multipole bins and
        different LR regions measured with the HM and DS splits. The undebiased
        $\ell=50-160$ bin is also listed for comparison to the noise debiased
        values. The quoted statistical uncertainties are one half of the region
        enclosing $95\%$ of the adjusted signal+noise simulations. The
        systematic uncertainties are the mean of the optimistic and pessimistic systematics estimates
        shown in Figure~\ref{fig:nh_curve}."""

        capPTE = r"""PTE statistic defined as the fraction of signal+noise
        simulations having $\rl^{BB}$ less than the observed value.
         PTEs do not account for systematic uncertainty."""


        f = open('figures/MLtable.tex','w')
        
        
        f.write(r'\newpage'+'\n')
        f.write(r'\begin{turnpage}'+'\n')

        f.write(r'\begin{table*}[tbp!] '+'\n')
        f.write(r'\centering '+'\n')
        f.write(r'\caption{'+capML+r'} '+'\n')
        f.write(r'\label{tab:ML} '+'\n')
        f.write(r'\begin{tabular}{lcccccccccc} '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\rule{0pt}{2ex} '+'\n')
        f.write(r'  & \mc{LR16}& \mc{LR24}& \mc{LR33}& \mc{LR42}& \mc{LR53}& \mc{LR63N}& \mc{LR63}& \mc{LR63S}& \mc{LR72} \\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'  $f_{\rm sky}^{\rm eff}$ [\%]\hfil& 16& 24& 33& 42& 53& 33& 63& 30& 72\\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\hline '+'\n')

        # EE/BB
        spec = 2 
        speclab = 'BB'

        binind = [0,0,0,1,2,3]

        types = ['cqu','cmc','cmcbk','cmcbk','cmcbk','cmcbk']
        systbin = [5, 5, 0, 1, 2, 3]

        bins = []
        for k in range(len(types)):
            be1 = getattr(self.c, types[k])['LR72'].be[binind[k]]
            be2 = getattr(self.c, types[k])['LR72'].be[binind[k]+1]
            bins.append('{:d}--{:d}'.format(be1,be2) )

        bins[0] = r'\begin{tabular}{@{}c@{}}'+bins[0]+r'\\ (no d.b.) \end{tabular}'

        ##########
        # M.L. R_BB
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'$\ell$ range & \multicolumn{9}{c}{Maximum Likelihood $\rl^{'+speclab+r'} (ML\substack{ +stat./syst.\\ -stat./syst.})$ \bigg( \begin{tabular}{@{}c@{}} HM \\[.1cm] DS \end{tabular} \bigg)   } \\')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')

        for j, binval, t, sb in zip(binind, bins, types, systbin):

            f.write(r' {0} '.format(binval))
            dat = getattr(self.c,t)
            datds = getattr(self.c,t+'ds')

            for k, val in enumerate(LR):

                R = dat[val].R[spec,j]
                up68 = dat[val].up68[spec,j] - R
                up95 = dat[val].up95[spec,j] - R
                down68 = R - dat[val].down68[spec,j]
                down95 = R - dat[val].down95[spec,j]

                Rds = datds[val].R[spec,j]
                up68ds = datds[val].up68[spec,j] - Rds
                up95ds = datds[val].up95[spec,j] - Rds
                down68ds = Rds - datds[val].down68[spec,j]
                down95ds = Rds - datds[val].down95[spec,j]

                be = dat[val].be[j:(j+2)]

                syst = 1-self.Rsystmean[sb][k][0]
                if syst>0:
                    systup = syst
                    systdown = 0
                else:
                    systup = 0
                    systdown = syst

                x = '${:0.3f}\substack{{ +{:0.2f}/{:0.2f}\\\\ -{:0.2f}/{:0.2f} }}$'.format(R,up95/2, systup, down95/2, systdown)
                y = '${:0.3f}\substack{{ +{:0.2f}/{:0.2f}\\\\ -{:0.2f}/{:0.2f} }}$'.format(Rds,up95ds/2,systup,down95ds/2,systdown)

                if ((up95/2)<.1) & ((down95/2)<.1):
                    err2 = ('{:.3f}'.format(up95/2)).replace('0.','.')
                    err4 = ('{:.3f}'.format(down95/2)).replace('0.','.')
                    systup = ('{:.3f}'.format(systup)).replace('0.','.')
                    systdown = ('{:.3f}'.format(systdown)).replace('0.','.')
                    x = '${:0.3f}\substack{{ +{:s}/{:s} \\\\ -{:s}/{:s} }}$'.format(R,err2,systup,err4,systdown)

                if ((up95ds/2)<.1) & ((down95ds/2)<.1):
                    err2 = ('{:.3f}'.format(up95ds/2)).replace('0.','.')
                    err4 = ('{:.3f}'.format(down95ds/2)).replace('0.','.')
                    if type(systup) is not str:
                        systup = ('{:.3f}'.format(systup)).replace('0.','.')
                        systdown = ('{:.3f}'.format(systdown)).replace('0.','.')
                    y = '${:0.3f}\substack{{ +{:s}/{:s} \\\\ -{:s}/{:s} }}$'.format(Rds,err2,systup,err4,systdown)

                if ~np.isfinite(Rds):
                    y = r'\ldots'


                if (binval=='20--55') & (val in ['LR16','LR24', 'LR33', 'LR42','LR53']):
                    f.write(r'& \ldots')
                else:
                    f.write(r'& \begin{tabular}{@{}c@{}}'+x+r' \\[.2cm]'+y+r'\end{tabular}')

            f.write(r'\\[1cm] '+'\n')
            f.write(r'\addlinespace[1ex] '+'\n')

            f.write(r'\addlinespace[1ex] '+'\n')


        f.write(r'\hline '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\end{tabular} '+'\n')
        f.write(r'\end{table*} '+'\n')
        f.write(r'\end{turnpage}'+'\n')
        f.close()

        ##########
        # PTE


        f = open('figures/PTEtable.tex','w')
        
        
        f.write(r'\begin{table*}[] '+'\n')
        f.write(r'\centering '+'\n')
        f.write(r'\caption{'+capPTE+r'} '+'\n')
        f.write(r'\label{tab:PTE} '+'\n')
        f.write(r'\begin{tabular*}{0.75\textwidth}{@{\extracolsep{\fill}} lcccccccccc} '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')

        f.write(r'\rule{0pt}{2ex} '+'\n')
        f.write(r'  & \mc{LR16}& \mc{LR24}& \mc{LR33}& \mc{LR42}& \mc{LR53}& \mc{LR63N}& \mc{LR63}& \mc{LR63S}& \mc{LR72} \\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'  $f_{\rm sky}^{\rm eff}$ [\%]\hfil& 16& 24& 33& 42& 53& 33& 63& 30& 72\\ '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r' & \multicolumn{9}{c}{PTE$_{'+speclab+r'}$ \Big( \begin{tabular}{@{}c@{}} HM \\[.05cm] DS \end{tabular} \Big)   } \\')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')
        f.write(r'\addlinespace[1ex] '+'\n')


        for j, binval, t in zip(binind, bins, types):

            f.write(r' {0} '.format(binval))
            dat = getattr(self.c,t)
            datds = getattr(self.c,t+'ds')

            for k, val in enumerate(LR):

                x = dat[val].PTE[spec,j]
                y = datds[val].PTE[spec,j]

                xx = r'{:0.3f}'.format(x)
                yy = r'{:0.3f}'.format(y)

                if ~np.isfinite(y):
                    yy = r'\ldots'

                if (binval=='20--55') & (val in ['LR16','LR24', 'LR33', 'LR42','LR53']):
                    f.write(r'& \ldots')
                else:
                    f.write(r'& \begin{tabular}{@{}c@{}}'+xx+r' \\'+yy+r'\end{tabular}')

            f.write(r'\\[.5cm] '+'\n')
            f.write(r'\addlinespace[1ex] '+'\n')


        f.write(r'\hline '+'\n')
        f.write(r'\hline '+'\n')
        f.write(r'\end{tabular*} '+'\n')
        f.write(r'\end{table*} '+'\n')

        f.close()



