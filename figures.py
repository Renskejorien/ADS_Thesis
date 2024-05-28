import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('City1.csv', sep=',', index=False)


# plot the results using matplotlib 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.phosphorus,
           df.utility,
           df.reliability)
ax.set_xlabel('Phosphorus')
ax.set_ylabel('Utility')
ax.set_zlabel('Reliability')
plt.show()