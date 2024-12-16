import numpy as np
import sys
import matplotlib.pyplot as plt
import data
from equations import *

np.set_printoptions(threshold=sys.maxsize)


# ------ I/O ------ # 

sigma = calculate_sigma(data.S,data.lamda)
#print(f'RCS={sigma}')
#print( to_db(sigma))

Pt = calculate_peak_power_track(data.SNR,data.B,data.lamda,data.Ts,data.L,data.R,data.G,sigma)
print(f'Pt={Pt}')




# ------ PLOTTING ------ # 


range_array = np.linspace(0,100000,1000) # Range values in [m]
Pt_values = []
for R in range_array:
    Pt_values.append(calculate_peak_power_track(data.SNR,data.B,data.lamda,data.Ts,data.L,R,data.G,sigma))


plt.plot(range_array,Pt_values)
plt.show()