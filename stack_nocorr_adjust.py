#! /usr/bin/env python

import sys
import os
import re

from datetime import datetime
from copy import deepcopy

import numpy as np

from lib import iolib
from lib import malib
from lib import timelib

stack_fn = sys.argv[1]
#offset = sys.argv[2]
#This is mean offset for PIG
#Note: PIG offset is trans_reference, so DEMs are above points!
offset = -3.1

s_orig = malib.DEMStack(stack_fn=stack_fn, stats=True, trend=True)
s = deepcopy(s_orig)

idx = np.array([bool(re.search('nocorr', i)) for i in s.source])
print 'Applying %0.2f m offset to %i nocorr DEMs' % (offset, np.nonzero(idx)[0].size)

#idx = [bool(re.search('nocorr', i)) for i in s.source]
#print 'Applying %0.2f m offset to %i nocorr DEMs' % (offset, sum(idx))

s.ma_stack[idx] += offset

out_stack_fn = os.path.splitext(sys.argv[1])[0]+'_nocorr_offset_%0.2fm.npz' % offset
s.stack_fn = out_stack_fn

save = True
if os.path.abspath(s_orig.stack_fn) == os.path.abspath(s.stack_fn):
    print "Warning: new stack has identical filename: %s" % s.stack_fn
    print "As a precaution, new stack will not be saved"
    save = False

#Update stats
if s.stats:
    s.compute_stats()
    if save:
        s.write_stats()
#Update trend 
if s.trend:
    s.compute_trend()
    if save:
        s.write_trend()
if save:
    s.savestack()
