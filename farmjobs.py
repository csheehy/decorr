import farmit
import numpy as np
import decorr

LR = decorr.getLR()
rlz = np.arange(100)
est = 'pspice'

sig = 'gaussian'
noi = 'qucov_noise'
pre = 'gaussian_'
for k,val in enumerate(LR):
    f = farmit.farmit('runjobs.py',args={'r':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':1})
    f.writejobfiles()

sig = 'gaussian'
noi = 'mcplusexcess_noise'
pre = 'gaussian_'
for k,val in enumerate(LR):
    f = farmit.farmit('runjobs.py',args={'r':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':1})
    f.writejobfiles()

if True:
    sig = 'pysm'
    noi = 'mcplusexcess_noise'
    pre = 'dust0_tophat_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'r':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':1})
        f.writejobfiles()

    sig = 'pysm'
    noi = 'mcplusexcess_noise'
    pre = 'dust1_tophat_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'r':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':1})
        f.writejobfiles()

    sig = 'pysm'
    noi = 'mcplusexcess_noise'
    pre = 'dust2_tophat_'
    for k,val in enumerate(LR):
        f = farmit.farmit('runjobs.py',args={'r':rlz, 'p':pre, 's':sig, 'n':noi, 'l':val, 'e':est}, reqs={'N':1})
        f.writejobfiles()

    f = farmit.farmit(f.jobfilepath+'*.job', resubmit=True)

f.runjobs(maxjobs=500)


