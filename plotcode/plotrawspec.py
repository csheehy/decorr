import decorr


for k,val in enumerate(decorr.getLR()):

    LR = val[2:]

    print val

    m = decorr.Model(fsky=ceil(float(LR[0:2])/10.)/10)
    doload = True
    ion()

    if doload:
        c = decorr.Calc('spec/gaussian/synch_therm_LR{:s}_mc_noise_xxxx.pickle'.format(LR),lmin=10,lmax=700,nbin=69,bintype='lin',full=False)
        l = c.bc

    #fac = l*(l+1)/(2*pi)
    fac = 1

    figure(1)
    clf()
    subplot(1,2,1)

    err = nanstd(c.sn,axis=0)
    col = ['b','r','g']
    lab = ['217x217', '353x353','217x353']

    for k in [0,2,1]:
        errorbar(l, c.r[k,2,:]/fac , err[k,2,:]/fac, fmt='.',color=col[k],label=lab[k])

    xexpec = sqrt(c.r[0,2,:]*c.r[1,2,:])
    plot(l, xexpec/fac, 'gx', label='expected cross given autos')

    d217 = m.getdustcl(l, 217, m.fsky, 2)/1e12*l*(l+1)/(2*pi)/fac
    d353 = m.getdustcl(l, 353, m.fsky, 2)/1e12*l*(l+1)/(2*pi)/fac
    plot(l, d217, 'b--',label='power law model')
    plot(l, d353, 'r--')
    plot(l, sqrt(d217*d353), 'g--')

    gca().set_yscale('log')
    ylim(1e-13,1e-9)
    xlim(0,250)
    title('real, LR'+LR)
    ylabel('l(l+1)Cl^BB/2pi')
    legend()
    grid('on')


    subplot(1,2,2)
    for k in [0,2,1]:
        errorbar(l, c.sn[1,k,2,:]/fac , err[k,2,:]/fac, fmt='.',color=col[k])

    xexpec = sqrt(c.sn[1,0,2,:]*c.r[1,2,:])
    plot(l, xexpec/fac, 'gx')

    plot(l, d217, 'b--')
    plot(l, d353, 'r--')
    plot(l, sqrt(d217*d353), 'g--')

    gca().set_yscale('log')
    ylim(1e-13,1e-9)
    xlim(0,250)
    title("sim realizaion 1, LR"+LR)
    ylabel('l(l+1)Cl^BB/2pi')
    grid('on')


    figure(2)
    clf()
    err = nanstd(c.SN,axis=0)
    plot(l,c.SN[:,2].T,alpha=0.5,color='gray',zorder=-100, label='sim realizatios')
    errorbar(l,c.R[2],err[2],fmt='.k', label='data')
    xlim(0,250)
    ylim(0.4,2)
    grid('on')
    ylabel('R')
    title('real, LR'+LR)

    figure(3)
    clf()
    err = nanstd(c.SN,axis=0)
    plot(l,c.SN[:,2].T,alpha=0.5,color='gray',zorder=-100, label='sim realizatios')
    errorbar(l,c.SN[1,2,:],err[2],fmt='.k', label='rlz 1')
    xlim(0,250)
    ylim(0.4,2)
    grid('on')
    ylabel('R')
    title('sim realization 1, LR'+LR)



    figure(1)
    savefig('finebin_LR{0}_spec.png'.format(LR))
    figure(2)
    savefig('finebin_LR{0}_R.png'.format(LR))
    figure(3)
    savefig('finebin_LR{0}_Rsim.png'.format(LR))

