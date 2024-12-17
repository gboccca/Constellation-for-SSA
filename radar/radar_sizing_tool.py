import numpy as np
import sys
import matplotlib.pyplot as plt
import data as data
import data_validation as dv
from equations import *

np.set_printoptions(threshold=sys.maxsize)


# ------ PLOTTING ------ # 
def plot_range_vs_peak_power(data):
    range_array = np.linspace(0, 100000, 1000)  # Range values in [m]
    Pt_values = []
    for R in range_array:
        Pt_values.append(calculate_peak_power_track(data, R))

    plt.plot(range_array, Pt_values, label='Peak Power vs Range')
    plt.xlabel('Range (m)')
    plt.ylabel('Peak Power (W)')
    plt.legend()

    # Add markers for ranges from 0 to 100000 m, every 10000m
    markers = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    for marker in markers:
        Pt_marker = calculate_peak_power_track(data, marker)
        plt.plot(marker, Pt_marker, 'ro')  # 'ro' means red color, circle marker
        plt.text(marker, Pt_marker, f'{Pt_marker:.2e} W', color='red', fontsize=8)  # Place text at the marker

    plt.show()

def plot_range_resolution_vs_B(data):
    range_array = np.linspace(10,50,1000) # Range resolution values in [m]
    B_values = []
    for R in range_array:
        B_values.append(range_resolution_to_bandwidth(R))

    plt.plot(B_values, range_array, label='Range Resolution vs Bandwidth')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Bandwidth (Hz)')
    plt.ylabel('Range Resolution (m)')
    plt.legend()

    # Add markers for range resolution values 10, 20, and 30m
    markers = [10, 20, 30, 40, 50]
    for marker in markers:
        B_marker = range_resolution_to_bandwidth(marker)
        plt.plot(B_marker, marker, 'ro')  # 'ro' means red color, circle marker
        plt.text(B_marker * 1.1, marker, f'{B_marker:.2e} Hz', color='red', fontsize=8)  # Shift text to the right

    plt.show()



# ------ I/O ------ # 

if __name__ == '__main__':
    Pt = calculate_peak_power_track(data)
    print(f'Pt@={Pt} @ R={data.R/1000}km')

    plot_range_vs_peak_power(data)
    plot_range_resolution_vs_B(data)
