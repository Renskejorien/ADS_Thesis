from platypus import Problem, Real, Hypervolume
from pyborg import BorgMOEA
import matplotlib.pyplot as plt
import numpy as np
from sys import *
from math import *
from scipy.optimize import root
import time
import datetime
 
# Lake Parameters
b = 0.42
q = 2.0

# Natural Inflow Parameters
mu = 0.02
sigma = 0.001

# Economic Benefit Parameters
alpha = 0.4 
delta = 0.98

# Set the number of RBFs (n), decision variables, objectives and constraints
nCities = 1
nvars = 9 * nCities
nobjs = 3
nYears = 100
nSamples = 100
nSeeds = 2
nconstrs = 1

# Set Thresholds
reliability_threshold = 0.85
inertia_threshold = -0.02

# Define the RBF Policy
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

    # Solve for the critical phosphorus level
    def pCrit(x):
        return [(x[0]**q) / (1 + x[0]**q) - b * x[0]]

    sol = root(pCrit, 0.5)
    critical_threshold = sol.x

    # Initialize arrays
    average_annual_P = np.zeros([nYears])
    discounted_benefit = np.zeros([nSamples])
    # yrs_inertia_met = np.zeros([nSamples])
    yrs_Pcrit_met = np.zeros([nSamples])
    lake_state = np.zeros([nYears + 1])
    objs = [0.0] * nobjs

    # Generate nSamples of nYears of natural phosphorus inflows
    natFlow = np.zeros([nSamples, nYears])
    for i in range(nSamples):
        np.random.seed(seed + i)
        natFlow[i, :] = np.random.lognormal(
            mean=log(mu**2 / np.sqrt(mu**2 + sigma**2)),
            sigma=np.sqrt(log((sigma**2 + mu**2) / mu**2)),
            size=nYears)

    # Determine centers, radii and weights of RBFs
    C = vars[0][0::3]
    R = vars[0][1::3]
    W = vars[0][2::3]

    # Normalize weights to sum to 1
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

    # Run model simulation
    for s in range(nSamples):
        lake_state[0] = 0
        Y = np.zeros([nYears])
        
        Y[0] = RBFpolicy(lake_state[0], C, R, normalize_W(W))

        for i in range(nYears):
            lake_state[i + 1] = lake_state[i] * (1 - b) + (lake_state[i]**q) / (1 + (lake_state[i]**q)) + Y[i] + natFlow[s, i]
            average_annual_P[i] = average_annual_P[i] + lake_state[i + 1] / nSamples
            discounted_benefit[s] = discounted_benefit[s] + alpha * Y[i] * delta**i

            # if i >= 1 and ((Y[i] - Y[i - 1]) > inertia_threshold):
            #     yrs_inertia_met[s] = yrs_inertia_met[s] + 1

            if lake_state[i + 1] < critical_threshold:
                yrs_Pcrit_met[s] = yrs_Pcrit_met[s] + 1

            if i < (nYears - 1):
                Y[i + 1] = RBFpolicy(lake_state[i + 1], C, R, newW)

    # Calculate minimization objectives
    objs[0] = np.max(average_annual_P)  # average annual P concentration
    objs[1] = -1 * np.sum(discounted_benefit) / nsamples # utility
    objs[2] = -1 * np.sum(yrs_pCrit_met) / (nYears * nSamples) # average reliability 

    return objs

# define the problem
problem = Problem(nvars, nobjs)
problem.types[0::3] = Real(-2, 2)
problem.types[1::3] = Real(0, 2)
problem.types[2::3] = Real(0, 1)
problem.function = LakeProblemDPS
 
# define and run the Borg algorithm for 10000 evaluations
algorithm = BorgMOEA(problem, epsilons = [0.01, 0.01, 0.0001])
# algorithm.run(10000)

def detailed_run(algorithm, maxevals, frequency, file):
    ''' Output runtime data for an algorithm run into a format readable by 
    the MOEAFramework library'''

    # open file and set up header
    f = open(file, "w+")
    f.write("# Variables = " + str(algorithm.problem.nvars))
    f.write("\n# Objectives = " + str(algorithm.problem.nobjs) + "\n")

    start_time = time.time()
    last_log = 0

    nvars = algorithm.problem.nvars
    nobjs = algorithm.problem.nobjs

    # run the algorithm/problem for specified number of function evaluations
    while (algorithm.nfe <= maxevals):
        # step the algorithm
        algorithm.step()

        # print to file if necessary
        if (algorithm.nfe >= last_log + frequency):
            last_log = algorithm.nfe
            f.write("#\n//ElapsedTime=" + str(datetime.timedelta(seconds=time.time()-start_time)))
            f.write("\n//NFE=" + str(algorithm.nfe) + "\n")
            arch = algorithm.archive[:]
            for i in range(len(arch)):
                sol = arch[i]
                for j in range(nvars):
                    f.write(str(sol.variables[j]) + " ")
                for j in range(nobjs):
                    f.write(str(sol.objectives[j]) + " ")
                f.write("\n")

    # close the runtime file
    f.close()

def runtime_hypervolume(algorithm):
    '''
    Calculate the hypervolume at a given frequency and build data
    arrays to plot.
    '''
    global last_calc
    if (algorithm.nfe >= last_calc + frequency):
        last_calc = algorithm.nfe
        nfe.append(last_calc)
        # use Platypus hypervolume indicator on the current archive
        result = hv.calculate(algorithm.archive[:])
        hyp.append(result)

# Define detailed_run parameters
maxevals = 100 
frequency = 5000 
hv = Hypervolume(minimum=[-1, 0, 0], maximum=[0, 3, 1]) 

# Run the algorithm
nfe, hyp = detailed_run(algorithm, maxevals, frequency, "city1.data", callback = runtime_hypervolume)

# plot the results using matplotlib 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([s.objectives[0] for s in algorithm.result],
           [s.objectives[1] for s in algorithm.result],
           [s.objectives[2] for s in algorithm.result])
ax.set_xlabel('Maximum Phosphorus')
ax.set_ylabel('Utility')
ax.set_zlabel('Reliability')
plt.show()

plt.plot(nfe, hyp)
plt.title('PyBorg Runtime Hypervolume 1 City')
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Hypervolume')
plt.show()
