
import numpy as np
from scipy import signal


def remove_dc(waveform, fs, gain=18, cutoff=70, filtfilt=True):
    wp = cutoff
    ws = wp / 2
    gpass = 1
    gstop = gain

    b, a = signal.iirdesign(wp, ws, gpass, gstop, ftype='cheby2', fs=fs)

    if filtfilt:
        ret = signal.filtfilt(b, a, waveform)
    else:
        raise NotImplementedError
    
    return ret