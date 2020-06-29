import cPickle as cP

f=open('spec/hmcorr/corr_hm1xhm2_217_xxxx.pickle')
s=cP.load(f)
f.close()
c217=decorr.Calc(s,bintype='lin',lmin=0,lmax=700,nbin=35)
c217.getR('sn');c217.getR('r')

f=open('spec/hmcorr/corr_hm1xhm2_353_xxxx.pickle')
s=cP.load(f)
f.close()
c353=decorr.Calc(s,bintype='lin',lmin=0,lmax=700,nbin=35)
c353.getR('sn');c353.getR('r')

err217=c217.SN.std(0)
err353=c353.SN.std(0)


clf()
l = c217.bc

subplot(1,2,1)
errorbar(l, c217.R[2] - c217.SN[:,2,:].mean(0), err217[2], fmt='.k')
plot([0,700],[0,0],':k')
title('217')

subplot(1,2,2)
errorbar(l, c353.R[2] - c353.SN[:,2,:].mean(0), err353[2], fmt='.k')
plot([0,700],[0,0],':k')
title('353')
