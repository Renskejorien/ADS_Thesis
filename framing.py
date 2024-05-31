# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import data
df2 = pd.read_csv('City2.csv', sep=',')
df2 = df2[df2['Reliability'] <= -0.3]
df3 = pd.read_csv('City3.csv', sep=',')
df3 = df3[df3['Reliability'] <= -0.3]

# Set ideal and anti-ideal values for each city
ideal2 = [min(df2['Utility1']), min(df2['Utility2'])]
antiideal2 = [max(df2['Utility1']), max(df2['Utility2'])]
ideal3 = [min(df3['Utility 1']), min(df3['Utility 2']), min(df3['Utility 3'])]
antiideal3 = [max(df3['Utility 1']), max(df3['Utility 2']), max(df3['Utility 3'])]

# Least Squares
def least_squares(df, ideal, nNegs, nSols, weight):
    LS_solutions = []
    for i in nNegs:
        neg = 0 
        sol = 0
        for j in nSols:    
            sol += weight * (ideal[neg] - df[i].iloc[j]) ** 2
        LS_solutions.append(sol)
        neg += 1
    LS = min(LS_solutions)
    LSindex = np.argmin(LS_solutions)
    return LS, LSindex

# MINIMAX
def minimax(df, ideal, nNegs, nSols, weight):
    MM = []
    MMindex = []
    for i in nNegs:
        neg = 0 
        sol = 0
        MM_solutions = []
        for j in nSols:    
            sol = weight * (ideal[neg] - df[i].iloc[j])
            MM_solutions.append(sol)
        MM.append(max(MM_solutions))
        MMindex.append(np.argmax(MM_solutions))
        neg += 1
    MINIMAX = min(MM)
    MINIMAXindex = np.argmin(MM)
    return MINIMAX, MINIMAXindex


# Compromise Programming
def compromise_programming(df, ideal, nNegs, nSols, weight, p, antiideal):
    CP_solutions = []
    for i in nNegs: 
        lp = 0
        neg = 0
        for j in nSols:
            lp += weight * abs((ideal[neg] - df[i].iloc[j]) / (ideal[neg] - antiideal[neg]) ** p)
        CP_solutions.append(lp ** (1 / p))
        neg += 1
    CP = min(CP_solutions)
    CPindex = np.argmin(CP_solutions)
    return CP, CPindex


# Power index
def normalize_power(p):
    normalized = np.zeros(len(p))
    total = sum(p)
    for i in range(len(p)):
        normalized[i] = p[i] / total
    return normalized

def power_index(df, ideal, nNegs, nSols, weight):
    PI = []
    for i in nNegs: 
        PI_solutions = []
        neg = 0
        denominator = 0
        for j in nSols:
            denominator += weight * (ideal[neg] - df[i].iloc[j])
        for j in nSols:
            Pneg = (weight * (ideal[neg] - df[i].iloc[j])) / denominator
            PI_solutions.append(Pneg)
        PI.append(normalize_power(PI_solutions))
    PIsum = max(np.sum(PI, axis=0))
    PIindex = np.argmax(np.sum(PI, axis=0))
    return PIsum, PIindex


# Set parameters for 2 cities
nNegs2 = df2.columns[0:2]
nSols2 = range(len(df2.index))
weight = 1

print('LS 2 cities: ', least_squares(df2, ideal2, nNegs2, nSols2, weight))
print('MM 2 cities: ', minimax(df2, ideal2, nNegs2, nSols2, weight))
print('CP 2 cities: ', compromise_programming(df2, ideal2, nNegs2, nSols2, weight, p=1, antiideal=antiideal2))
print('PI 2 cities: ', power_index(df2, ideal2, nNegs2, nSols2, weight))

print('\n')

# Set parameters for 3 cities
nNegs3 = df3.columns[0:3]
nSols3 = range(len(df3.index))
weight = 1

print('LS 3 cities: ', least_squares(df3, ideal3, nNegs3, nSols3, weight))
print('MM 3 cities: ', minimax(df3, ideal3, nNegs3, nSols3, weight))
print('CP 3 cities: ', compromise_programming(df3, ideal3, nNegs3, nSols3, weight, p=1, antiideal=antiideal3))
print('PI 3 cities: ', power_index(df3, ideal3, nNegs3, nSols3, weight))
