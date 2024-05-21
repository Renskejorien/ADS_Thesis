from platypus import Problem, Real, Hypervolume
from pyborg import BorgMOEA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sys import *
from math import *
from scipy.optimize import root
import scipy.stats as ss
 
# Lake Parameters
b = 0.42
q = 2.0

# Natural Inflow Parameters
mu = 0.03
sigma = np.sqrt(10**-5)

# Economic Benefit Parameters
alpha = 0.4 
delta = 0.98

# Set the number of RBFs (n), decision variables, objectives and constraints
nCities = 2
nvars = 9 * nCities
nobjs = 4
nYears = 100
nSamples = 100
nSeeds = 2
nconstrs = 1

# Set Thresholds
reliability_threshold = 0.85
inertia_threshold = -0.02

#Define the RBF Policy
def RBFpolicy(lake_state, C, R, W):
    # Determine pollution emission decision, Y
    Y = 0
    for i in range(len(C)):
        if R[i] != 0:
            Y = Y + W[i] * ((np.absolute(lake_state - C[i]) / R[i])**3)

    Y = min(0.1, max(Y, 0.01))

    return Y

def LakeProblemDPS(*vars):
    seed = 1234

    #Solve for the critical phosphorus level
    def pCrit(x):
        return [(x[0]**q) / (1 + x[0]**q) - b * x[0]]

    sol = root(pCrit, 0.5)
    critical_threshold = sol.x

    #Initialize arrays
    average_annual_P = np.zeros([nYears])
    discounted_benefit = np.zeros([nSamples])
    yrs_inertia_met = np.zeros([nSamples])
    yrs_Pcrit_met = np.zeros([nSamples])
    lake_state = np.zeros([nYears + 1])
    objs = [0.0] * nobjs
    constrs = [0.0] * nconstrs
    reliability = 0.0

    #Generate nSamples of nYears of natural phosphorus inflows
    natFlow = np.zeros([nSamples, nYears])
    for i in range(nSamples):
        np.random.seed(seed + i)
        natFlow[i, :] = np.random.lognormal(
            mean=log(mu**2 / np.sqrt(mu**2 + sigma**2)),
            sigma=np.sqrt(log((sigma**2 + mu**2) / mu**2)),
            size=nYears)

    # Determine centers, radii and weights of RBFs
    C1 = vars[0][0::6]
    R1 = vars[0][1::6]
    W1 = vars[0][2::6]
    C2 = vars[0][3::6]
    R2 = vars[0][4::6]
    W2 = vars[0][5::6]

    #Normalize weights to sum to 1
    def normalize_W(W):
        newW = np.zeros(len(W))
        total = sum(W)
        if total != 0.0:
            for i in range(len(W)):
                newW[i] = W[i] / total
        else:
            for i in range(len(W)):
                newW[i] = 1 / 3
        return newW

    #Run model simulation
    for s in range(nSamples):
        lake_state[0] = 0
        Y1 = np.zeros([nYears])
        Y2 = np.zeros([nYears])
        
        Y1[0] = RBFpolicy(lake_state[0], C1, R1, normalize_W(W1))
        Y2[0] = RBFpolicy(lake_state[0], C2, R2, normalize_W(W2))

        for i in range(nYears):
            lake_state[i + 1] = lake_state[i] * (1 - b) + (
                lake_state[i]**q) / (1 + (lake_state[i]**q)) + Y1[i] + Y2[i] + natFlow[s, i]
            average_annual_P[i] = average_annual_P[i] + lake_state[i + 1] / nSamples
            discounted_benefit[s] = discounted_benefit[s] + alpha * Y1[i] * delta**i + alpha * Y2[i] * delta**i

            if i >= 1 and (((Y1[i] - Y1[i - 1]) + (Y2[i] - Y2[i - 1])) > inertia_threshold):
                yrs_inertia_met[s] = yrs_inertia_met[s] + 1

            if lake_state[i + 1] < critical_threshold:
                yrs_Pcrit_met[s] = yrs_Pcrit_met[s] + 1

            if i < (nYears - 1):
                #find policy-derived emission
                Y1[i + 1] = RBFpolicy(lake_state[i + 1], C1, R1, normalize_W(W1))
                Y2[i + 1] = RBFpolicy(lake_state[i + 1], C2, R2, normalize_W(W2))
        reliability += np.sum(lake_state < root(pCrit, 0.5).x)/float(nSamples*nvars)

    # Calculate minimization objectives (defined in comments at beginning of file)
    objs[0] = np.max(average_annual_P)  #minimize the max average annual P concentration
    objs[1] = np.sum(delta * (alpha * Y1[0] - b * average_annual_P**2)) #utility1
    objs[2] = np.sum(delta * (alpha * Y2[0] - b * average_annual_P**2)) #utility2
    objs[3] = reliability  #reliability

    constrs[0] = max(0.0, reliability_threshold - (-1 * objs[2])) # is this necessary?
    return objs

# define the problem
problem = Problem(nvars, nobjs)
problem.types[0::6] = Real(-2, 2)
problem.types[1::6] = Real(0, 2)
problem.types[2::6] = Real(0, 1)
problem.types[3::6] = Real(-2, 2)
problem.types[4::6] = Real(0, 2)
problem.types[5::6] = Real(0, 1)
problem.function = LakeProblemDPS
 
# define and run the Borg algorithm for 10000 evaluations
algorithm = BorgMOEA(problem, epsilons=[0.01, 0.01, 0.01, 0.0001])
algorithm.run(10000)
 
# plot the results using matplotlib 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')
plt.show()
