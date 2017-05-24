import os
import string
import random
import numpy as np
from glob import glob
from datetime import datetime

class farmit(object):

    def __init__(self, script, args=None, reqs=None, jobname=None, resubmit=False):
        """e.g. 
        f=farmit.farmit('test.py', jobname='test', args={'fmin':[1,2],
        'fmax':[5,6]}, reqs={'N':2})
        f.writejobfiles()
        f.runjobs()

        Values of argument dictonary must be arrays, even if only length 1
        """

        if resubmit:
            #First arg is wildcard to existing job files. Just use these and do
            #nothing else
            self.jobfilenames = glob(script)
            return

        self.script = script
        self.args = args
        self.reqs = reqs

        if jobname is None:
            self.jobname = self.datestr()+'_'+self.randstring()
        else:
            self.jobname = jobname
        
        self.prepargs()

        self.jobfilepath = os.getenv('HOME')+'/jobfiles/'
        self.getjobfilenames()
        self.runpath = os.getcwd()

        return

    def prepargs(self):
        """Make arguments all numpy arrays"""
        if self.args is not None:
            nvals = []
            for k,val in enumerate(self.args):
                x = self.args[val]
                if np.size(x) == 1:
                    x = np.reshape(np.array(x),(1))
                    self.args[val] = x
                    nvals.append(1)
                else:
                    x = np.array(x)
                    self.args[val] = x
                    nvals.append(x.size)
            self.njobs = np.max(np.array(nvals))
        else:
            self.njobs = 1

    def getjobfilenames(self):
        """Get job file names, sets self.jobfilename"""
        self.jobfilenames = []
        for k in range(self.njobs):
            fn = self.jobname + '_' + '{:04d}'.format(k) + '.job'
            self.jobfilenames.append(self.jobfilepath + fn)


    def randstring(self, size=4):
        """Generate random string of size size"""
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choice(chars) for _ in range(size))


    def datestr(self):
        """Get current date and time"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')


    def getcmd(self, i=0):
        """Get command line command to issue. Args can come as an array of
        values, so use the ith value"""
        cmd = []
        cmd.append('cd ' + self.runpath)

        cmd0 = 'python '
        cmd0 += self.script
        if self.args is not None:
            for k,val in enumerate(self.args):
                cmd0 += ' -'
                cmd0 += val
                cmd0 += ' '
                if self.args[val].size > 1:
                    cmd0 += str(self.args[val][i])
                else:
                    cmd0 += str(self.args[val][0])

        cmd.append(cmd0)

        cmd.append('bash -c "if [ $? == "0" ]; then rm -f {0}; fi"'.format(self.jobfilenames[i]))

        return cmd


    def writejobfiles(self):
        """Write job files to $HOME/jobfiles/"""
        for k,val in enumerate(self.jobfilenames):

            f = open(val, 'w')
            f.write('command: |\n')
            cmd = self.getcmd(k)

            for j,cmd0 in enumerate(cmd):
                f.write('    ' + cmd0+'\n')

            if self.reqs is not None:
                for j,req in enumerate(self.reqs):
                    f.write(self.reqstring(req))
            f.close()

    def reqstring(self, req):
        return '{0}: {1}\n'.format(req, self.reqs[req])

    def runjobs(self):
        """Submit jobs"""
        for k,val in enumerate(self.jobfilenames):
            cmd = 'wq sub -b'
            cmd += ' '
            cmd += val
            os.system(cmd)
