import numpy as np

array1 = np.random.rand(1, 3)  # Shape (1,3)
array2 = np.random.rand(14, 3) # Shape (14,3)

concatenated = np.concatenate((array1, array2), axis=0)

print(concatenated.shape)  # Output: (15, 3)