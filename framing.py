# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import data
df4 = pd.read_csv('City2.csv', sep=',')
df2 = df4[df4['Reliability'] >= 0.85]
df5 = df4[df4['Reliability'] >= 0.95]
df6 = df4[df4['Reliability'] >= 0.99]
df3 = pd.read_csv('City3.csv', sep=',')
df3 = df3[df3['Reliability'] <= -0.3]

print(df5.head())
# Set ideal and anti-ideal values for each city
ideal2 = [max(df2['Utility1']), max(df2['Utility2'])]
ideal4 = [max(df4['Utility1']), max(df4['Utility2']), max(df4['normalized_reliability'])]
ideal3 = [max(df3['Utility 1']), max(df3['Utility 2']), max(df3['Utility 3'])]

# Utilitarian
def utilitarian(df):
    U_solutions = df['Utility1'] + df['Utility2']
    U = max(U_solutions)
    Uindex = np.argmax(U_solutions)
    return U, Uindex

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

def fallback_bargaining(df, i, j, k):
	# solution numbers for indexing
	sol_nums = df.iloc[:, 1]

	# sort solutions by satisficing for each utility, largest to smallest
	U1_sorted = np.argsort(df.iloc[:, i])[::-1]
	U2_sorted = np.argsort(df.iloc[:, j])[::-1]
	if k > 0:
		R_sorted = np.argsort(df.iloc[:, k])[::-1]

	# rank solutions for each variable
	U1_ranking = U1_sorted[sol_nums]
	U2_ranking = U2_sorted[sol_nums]
	if k > 0:
		R_ranking = R_sorted[sol_nums]

	# initialize proposals
	U1_proposed = np.array([])
	U2_proposed = np.array([])
	if k > 0:
		R_proposed = np.array([])

	compromise = np.array([])
	l = 0

	# perform fallback bargaining
	while len(compromise) < 1:
		if l in sol_nums:
			# get the ith preference of each utility
			U1_proposed = np.append(U1_proposed, U1_ranking[l])
			U2_proposed = np.append(U2_proposed, U2_ranking[l])
			if k > 0:
				R_proposed = np.append(R_proposed, R_ranking[l])

			# check to see if there is a common solution across utilities
			U_intersect = np.intersect1d(U1_proposed, U2_proposed)
			if len(U_intersect) > 0:
				if k > 0:
					compromise = np.intersect1d(U_intersect, R_proposed)
					if len(compromise) > 0:
						return compromise[0]
				else: 
					if len(U_intersect) > 0:
						return U_intersect[0]
		l += 1

          
# Set parameters for 2 cities
nNegs2 = df2.columns[1:3]
nSols2 = range(len(df2.index))
weight = 1

print('U 2 cities R85: ', utilitarian(df2))
print('LS 2 cities R85: ', least_squares(df2, ideal2, nNegs2, nSols2, weight))
print('MM 2 cities R85: ', minimax(df2, ideal2, nNegs2, nSols2, weight))
print('PI 2 cities R85: ', power_index(df2, ideal2, nNegs2, nSols2, weight))
print('FB 2 city R85: ', fallback_bargaining(df2, 3, 4, 0))

print('\n')

nSols5 = range(len(df5.index))
print('U 2 cities R95: ', utilitarian(df5))
print('LS 2 cities R95: ', least_squares(df5, ideal2, nNegs2, nSols5, weight))
print('MM 2 cities R95: ', minimax(df5, ideal2, nNegs2, nSols5, weight))
print('PI 2 cities R95: ', power_index(df5, ideal2, nNegs2, nSols5, weight))
print('FB 2 city R95: ', fallback_bargaining(df5, 3, 4, 0))

print('\n')

nSols6 = range(len(df6.index))
print('U 2 cities R99: ', utilitarian(df6))
print('LS 2 cities R99: ', least_squares(df6, ideal2, nNegs2, nSols6, weight))
print('MM 2 cities R99: ', minimax(df6, ideal2, nNegs2, nSols6, weight))
print('PI 2 cities R99: ', power_index(df6, ideal2, nNegs2, nSols6, weight))
print('FB 2 city R99: ', fallback_bargaining(df6, 3, 4, 0))

print('\n')

# Set parameters for 2 cities with reliability as a third party
nNegs2r = ["Utility1", "Utility2", "normalized_reliability"]
nSols2r = range(len(df4.index))
weight = 1

print('U 2 cities + reliability: ', utilitarian(df4))
print('LS 2 cities + reliability: ', least_squares(df4, ideal2, nNegs2r, nSols2r, weight))
print('MM 2 cities + reliability: ', minimax(df4, ideal2, nNegs2r, nSols2r, weight))
print('PI 2 cities + reliability: ', power_index(df4, ideal2, nNegs2r, nSols2r, weight))
print('FB 2 city + reliability: ', fallback_bargaining(df4, 3, 4, 6))

print('\n')

# Set parameters for 3 cities
nNegs3 = df3.columns[0:3]
nSols3 = range(len(df3.index))
weight = 1

print('LS 3 cities: ', least_squares(df3, ideal3, nNegs3, nSols3, weight))
print('MM 3 cities: ', minimax(df3, ideal3, nNegs3, nSols3, weight))
print('PI 3 cities: ', power_index(df3, ideal3, nNegs3, nSols3, weight))
