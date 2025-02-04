import matplotlib.pyplot as plt
import numpy as np  

''''

this code plots the gain vs N obtained as a result from running the gain simulator. 
again, this is tupid because we know that gain is a linear function of N. If anything, it works as validation of the gain pattern calculator.
Remember to do more validation tho.


'''
# Define N and Gain values based on the runs
N_values = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400,
            484, 529, 625, 676, 784, 900, 1024, 1156, 1296, 1444, 1681, 1849, 2116, 2401, 2704, 3136,
            3481, 3969, 4489, 5041, 5776, 6561, 7396, 8464, 9604, 10816, 12321, 13924, 15876, 17956,
            20164, 23104, 26244, 29584, 33489, 38025, 43264, 48841, 55225, 63001, 71289, 80656, 91809, 
            103684, 117649, 133225, 151321, 171396, 194481, 220900, 250000]

Gain_values = [4.97, 10.99, 14.51, 17.01, 18.95, 20.53, 21.87, 23.03, 24.05, 24.97, 25.79, 26.55, 27.25, 
               27.89, 28.49, 29.05, 29.58, 30.07, 30.54, 30.99, 31.82, 32.21, 32.93, 33.27, 33.91, 34.51, 
               35.07, 35.6, 36.1, 36.57, 37.22, 37.64, 38.23, 38.77, 39.29, 39.93, 40.39, 40.96, 41.49, 42.0, 
               42.59, 43.14, 43.66, 44.25, 44.79, 45.31, 45.88, 46.41, 46.98, 47.51, 48.02, 48.61, 49.16, 
               49.68, 50.22, 50.77, 51.33, 51.86, 52.39, 52.96, 53.5, 54.03, 54.6, 55.13, 55.68, 56.22, 
               56.77, 57.31, 57.86, 58.41, 58.95]

Gain_values = [10**(gain/10) for gain in Gain_values]
#N_values = [10* np.log10(N) for N in N_values]
# Plot N vs Gain with markers at each data point
plt.figure(figsize=(10, 6))
plt.plot(N_values, Gain_values, linestyle='-', color='b', marker='o', markersize=4)

# Add labels and title
plt.xlabel('N [-]')
plt.ylabel('Gain [-]')
plt.title('N vs Gain')
plt.grid(True)

# Show the plot
plt.show()
