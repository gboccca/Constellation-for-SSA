import numpy as np
# ------ constants ------ # 
k = 1.38e-23    # Boltzmann []


# ------ design data ------ # 

SNR = 10        # Signal-to-Noise Ratio [-]
B = 1e6           # Bandwidth [Hz]
lamda = 0.006   # Wavelength [m]
eta = 0.3       # Antenna Efficiency [-]
S = 0.0115      # Minimum Detectable debris diameter [m]
N = 10000       # Number of array elements
A = N*0.003**2  # Antenna Aperture (Effective Area) [m2]
d = lamda*0.5   # Distance between array elements [m]
G = 10**(40/10)       # Antenna Gain [-] - from gainpattern.py


# ------ performance data ------ # 

Ts = 120        # System Noise Temperature [K] assuming mattia receivers become smaller in future years
L = 3           # Losses [-]
R = 20000       # Range [m]


# ------ RCS calculation ------ #

def calculate_sigma(S,lamda):

    sigma = S**2 * np.pi/4  
    print(sigma/(lamda**2))
    if sigma/(lamda**2) > 2.835:
        return sigma
    elif sigma/(lamda**2)< 0.00122: 
        sigma = S**2 * 9/4 * np.pi**5
        return sigma
    else:
        raise ValueError('Resonance wavelength selected')

RCS = calculate_sigma(S,lamda)