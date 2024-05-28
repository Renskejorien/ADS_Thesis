import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('City1.csv', sep=',', index=False)
ideal = [2, -2, -1] # ideal solution [phosphorus, utility, reliability]
players = 2
weight = 1

# Least Squares
LS_solutions = [0.0 for i in range(len(df.index))]
for i in range(len(df.index)): 
    square = weight * (ideal - i) ** 2
    LS_solutions.append(i)
LS = min(sum(LS_solutions))

# MINIMAX
MM_solutions = [0.0 for i in range(len(df.index))]
for i in range(len(df.index)): 
    square = weight * (ideal - i) ** 2
    MM_solutions.append(i)
MM = min(max(MM_solutions))

# Compromise Programming
CP_solutions = [0.0 for i in range(len(df.index))]
for i in range(len(df.index)): 
    square = weight * (ideal - i) ** 2
    CP_solutions.append(i)
CP = min(max(CP_solutions))

# Power index
PI_solutions = [0.0 for i in range(len(df.index))]
for i in range(len(df.index)): 
    PI_solutions[i] = (weight * (ideal - i)) / sum(weight * (ideal - i))

def normalize_power(p):
    np = np.zeros(len(p))
    total = sum(p)
    for i in range(len(p)):
        np[i] = p[i] / total
    return np
PI = normalize_power(PI_solutions)