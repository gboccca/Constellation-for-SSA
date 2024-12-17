import numpy as np
from data import k

# ------ DB CONVERTER ------ # 

def to_db (value):
    return 10*np.log10(value)

def from_db (value):
    return 10**(value/10)

# ------ RADAR EQUATIONS ------ #

def calculate_peak_power_track(data,R=None):
    '''Calculates the peak power required to track a target at a given range R. If R is not provided, the maximum range is taken from the data object.'''	
    if R is None:
        R = data.R
    Pt_track = R**4*(4*np.pi)**3*data.SNR*k*data.Ts*data.B*data.L/(data.G**2*data.lamda**2*data.RCS)
    return Pt_track


def bandwidth_to_range_resolution(B):
    return 3e8/(2*B)

def range_resolution_to_bandwidth(R):
    return 3e8/(2*R)