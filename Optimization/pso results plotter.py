from matplotlib import pyplot as plt
import numpy as np


run_name = 'run2'
filename = f'Optimization/pso_results/{run_name}.csv'

with open(filename, 'r') as file:
    data = file.readlines()


pso_history = data[11]


print(pso_history)

