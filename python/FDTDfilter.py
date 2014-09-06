# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:11:34 2014

@author: Jukka Saarelma
"""

import numpy as np

from scipy.signal import kaiserord, lfilter, firwin


def FDTDfilter(x, sfs, fs, normalized_cutoff):
  nyq_rate = sfs / 2.0
  ripple_db = 70.0
  cutoff_hz = sfs*normalized_cutoff
  taps = firwin(200, cutoff_hz/nyq_rate, window=('chebwin', ripple_db))
  filtered_x = lfilter(taps, 1.0, x, axis =0)
  return filtered_x

