import numpy as np
# ------ constants ------ # 
k = 1.38e-23    # Boltzmann []


# ------ design data ------ # 

SNR = 10        # Signal-to-Noise Ratio [-]
S = 0.01      # Minimum Detectable debris diameter [m]
B = 1e6           # Bandwidth [Hz]
lamda = np.pi*S  # Wavelength [m]
eta = 0.3       # Antenna Efficiency [-]
N = 250000       # Number of array elements
A = N*(lamda/2)**2  # Antenna Aperture (Effective Area) [m2]
d = lamda*0.5   # Distance between array elements [m]
G = 10**(50/10)       # Antenna Gain [-] - from gainpattern.py


# ------ performance data ------ # 

Ts = 120        # System Noise Temperature [K] assuming mattia receivers become smaller in future years
L = 3           # Losses [-]
R = 20000       # Range [m]


# ------ RCS calculation ------ #

def calculate_sigma(S,lamda):

    if np.pi*S/lamda <= 1:
        sigma = S**2 * 9/4 * np.pi**5

    elif np.pi*S/lamda >= 10:

        sigma = S**2 * np.pi/4  

    else:
        raise ValueError('Resonance wavelength selected')
    
    return sigma


RCS = calculate_sigma(S,lamda)
print(f'RCS = {RCS:.2e} m2')