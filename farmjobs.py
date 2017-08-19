import farmit
import numpy as np
import decorr

#LR = decorr.getLR()
LR = ['LR72']
rlz = np.arange(100)
est = 'pspice'
N = 1

if False:
    sig = 'gaussian'
    noi = 'qucovds_noise'
    pre = 'gaussian_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'z':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
        f.writejobfiles()

    sig = 'gaussian'
    noi = 'qucov_noise'
    pre = 'gaussian_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'z':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
        f.writejobfiles()

    sig = 'gaussian'
    noi = 'mcplusexcess_noise'
    pre = 'gaussian_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'z':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
        f.writejobfiles()

    sig = 'pysm'
    noi = 'mcplusexcess_noise'
    pre = 'dust0_tophat_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'z':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
        f.writejobfiles()

    sig = 'pysm'
    noi = 'mcplusexcess_noise'
    pre = 'dust1_tophat_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'z':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
        f.writejobfiles()

    sig = 'pysm'
    noi = 'mcplusexcess_noise'
    pre = 'dust2_tophat_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'z':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
        f.writejobfiles()

sig = 'pysm'
noi = 'noihmds1'
pre = 'dust0_tophat_'
r = 'realhmds1'
for k,val in enumerate(LR):
    f = farmit.farmit('runjobs.py',args={'z':rlz, 'r':r, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
    f.writejobfiles()

sig = 'pysm'
noi = 'noihmds2'
pre = 'dust0_tophat_'
r = 'realhmds2'
for k,val in enumerate(LR):
    f = farmit.farmit('runjobs.py',args={'z':rlz, 'r':r, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
    f.writejobfiles()

sig = 'pysm'
noi = 'noihmdsavg'
pre = 'dust0_tophat_'
r = 'realhmdsavg'
for k,val in enumerate(LR):
    f = farmit.farmit('runjobs.py',args={'z':rlz, 'r':r, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':N})
    f.writejobfiles()


f = farmit.farmit(f.jobfilepath+'*.job', resubmit=True)
f.runjobs(maxjobs=500)


