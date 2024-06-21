import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../City2.csv', sep=',')
data['Reliability_99'] = np.where((data['Reliability'] >= .99), 'Reliability ≥ .99', 'Reliability < .99') # column 14
data['Reliability_95'] = np.where((data['Reliability'] >= .95), 'Reliability ≥ .95', 'Reliability < .95') # column 15
data['Reliability_85'] = np.where((data['Reliability'] >= .85), 'Reliability ≥ .85', 'Reliability < .85') # column 16
data_85 = data.loc[data['Reliability_85'] == 'Reliability ≥ .85']
data_95 = data.loc[data['Reliability_95'] == 'Reliability ≥ .95']
data_99 = data.loc[data['Reliability_99'] == 'Reliability ≥ .99']
print(data_99.head)

# Data >= 85
def conditions(vals):
    if   vals == 61 or vals == 45 or vals == 92:   
        return "Utilitarian"
    elif vals == 1 or vals == 2:   
        return "Social Planner"
    elif vals == 0:   
        return "Egalitarian"
    elif vals == 113 or vals == 103 or vals == 95:   
        return "Pragmatic - Power Index"
    elif vals == 25 or vals == 47 or vals == 44 or vals == 2151:   
        return "Pragmatic - Fallback Bargaining"
    else:               
        return "Non-equitable solution"

data["Optimal Solutions"] = data["Unnamed: 0"].map(conditions)
data["Solution"] = 'Non-equitable solution'
print(data.iloc[:, [17]].head)
parallel_coordinates(data.iloc[:, [2,3,4,5,17]], 'Solution', color = '#D8D8D8')
parallel_coordinates(data.iloc[[61, 1, 0, 113, 25], [2,3,4,5,16]], 'Optimal Solutions', color = ['#FFA500', '#2986cc', '#8fce00', '#f44336', '#c90076', '#bcbcbc'])
plt.show()