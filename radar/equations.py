import numpy as np
from data import k

# ------ DB CONVERTER ------ # 

def to_db (value):
    return 10*np.log10(value)

def from_db (value):
    return 10**(value/10)


# ------ CALCULATOR ------ # 


def calculate_sigma(S,lamda):

    sigma = S**2 * np.pi/4  
    print(sigma/(lamda**2))
    if sigma/(lamda**2) > 2.835:
        return sigma
    elif sigma/(lamda**2)< 0.00122: 
        sigma = S**2 * 9/4 * np.pi**5
        return sigma
    else:
        print ('resonance in RCS!')




def calculate_peak_power_track(SNR,B,lamda,Ts,L,R,G,sigma):
    Pt_track = R**4*(4*np.pi)**3*SNR*k*Ts*B*L/(G**2*lamda**2*sigma)
    return Pt_track