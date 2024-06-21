import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df2 = pd.read_csv('../City2.csv', sep=',')

u_sol = [(row['Utility1'], row['Utility2'], row['Reliability']) for index, row in df2.loc[[61, 45, 92, 0]].iterrows()]
ls_sol = [(row['Utility1'], row['Utility2'], row['Reliability']) for index, row in df2.loc[[1, 1, 1, 1]].iterrows()]
mm_sol = [(row['Utility1'], row['Utility2'], row['Reliability']) for index, row in df2.loc[[0, 0, 0, 0]].iterrows()]
pi_sol = [(row['Utility1'], row['Utility2'], row['Reliability']) for index, row in df2.loc[[113, 103, 95, 0]].iterrows()]
fb_sol = [(row['Utility1'], row['Utility2'], row['Reliability']) for index, row in df2.loc[[25, 47, 44, 2151]].iterrows()]
df_sol = pd.DataFrame([u_sol, ls_sol, mm_sol, pi_sol, fb_sol], 
                      columns = ['85', '95', '99', '3 actor'],
                      index = ['u', 'ls', 'mm', 'pi', 'fb'])

df2 = df2.drop([61, 45, 92, 0, 1, 113, 103, 95, 25, 47, 44, 2151])

# Plot Reliability <= .85 
fig = plt.figure()
ax = fig.add_subplot(231, projection='3d')
ax.scatter(df2['Utility1'],
           df2['Utility2'],
           df2['Reliability'],
           color = '#D8D8D8', alpha = 0.1)
ax.scatter(df_sol.loc['u', '85'][0], df_sol.loc['u', '85'][1], df_sol.loc['u', '85'][2], 
        color='r', marker='o', alpha=1.0, label = 'U 85')
ax.scatter(df_sol.loc['ls', '85'][0], df_sol.loc['ls', '85'][1], df_sol.loc['ls', '85'][2], 
        color='y', marker='o', alpha=1.0, label = 'LS 85')
ax.scatter(df_sol.loc['mm', '85'][0], df_sol.loc['mm', '85'][1], df_sol.loc['mm', '85'][2], 
        color='g', marker='o', alpha=1.0, label = 'MM 85')
ax.scatter(df_sol.loc['pi', '85'][0], df_sol.loc['pi', '85'][1], df_sol.loc['pi', '85'][2], 
        color='b', marker='o', alpha=1.0, label = 'PI 85')
ax.scatter(df_sol.loc['fb', '85'][0], df_sol.loc['fb', '85'][1], df_sol.loc['fb', '85'][2], 
        color='c', marker='o', alpha=1.0, label = 'FB 85')
ax.set_xlabel('Utility 1')
ax.set_ylabel('Utility 2')
ax.set_zlabel('Reliability')
ax.set_title('Reliability >= .85', y = 1.0, pad = 0, fontsize = 8)

ax = fig.add_subplot(232, projection='3d')
ax.scatter(df2['Utility1'],
           df2['Utility2'],
           df2['Reliability'],
           color = '#D8D8D8', alpha = 0.1)
ax.scatter(df_sol.loc['u', '95'][0], df_sol.loc['u', '95'][1], df_sol.loc['u', '95'][2], 
        color='r', marker='o', alpha=1.0, label = 'U 95')
ax.scatter(df_sol.loc['ls', '95'][0], df_sol.loc['ls', '95'][1], df_sol.loc['ls', '95'][2], 
        color='y', marker='o', alpha=1.0, label = 'LS 95')
ax.scatter(df_sol.loc['mm', '95'][0], df_sol.loc['mm', '95'][1], df_sol.loc['mm', '95'][2], 
        color='g', marker='o', alpha=1.0, label = 'MM 95')
ax.scatter(df_sol.loc['pi', '95'][0], df_sol.loc['pi', '95'][1], df_sol.loc['pi', '95'][2], 
        color='b', marker='o', alpha=1.0, label = 'PI 95')
ax.scatter(df_sol.loc['fb', '95'][0], df_sol.loc['fb', '95'][1], df_sol.loc['fb', '95'][2], 
        color='c', marker='o', alpha=1.0, label = 'Reliability >= 95')
ax.set_xlabel('Utility 1')
ax.set_ylabel('Utility 2')
ax.set_zlabel('Reliability')
ax.set_title('Reliability >= .95', y = 1.0, pad = 0, fontsize = 8)

ax = fig.add_subplot(234, projection='3d')
ax.scatter(df2['Utility1'],
           df2['Utility2'],
           df2['Reliability'],
           color = '#D8D8D8', alpha = 0.1)
ax.scatter(df_sol.loc['u', '99'][0], df_sol.loc['u', '99'][1], df_sol.loc['u', '99'][2], 
        color='r', marker='o', alpha=1.0, label = 'U 99')
ax.scatter(df_sol.loc['ls', '99'][0], df_sol.loc['ls', '99'][1], df_sol.loc['ls', '99'][2], 
        color='y', marker='o', alpha=1.0, label = 'LS 99')
ax.scatter(df_sol.loc['mm', '99'][0], df_sol.loc['mm', '99'][1], df_sol.loc['mm', '99'][2], 
        color='g', marker='o', alpha=1.0, label = 'MM 99')
ax.scatter(df_sol.loc['pi', '99'][0], df_sol.loc['pi', '99'][1], df_sol.loc['pi', '99'][2], 
        color='b', marker='o', alpha=1.0, label = 'PI 99')
ax.scatter(df_sol.loc['fb', '99'][0], df_sol.loc['fb', '99'][1], df_sol.loc['fb', '99'][2], 
        color='c', marker='o', alpha=1.0, label = 'Reliability >= 99')
ax.set_xlabel('Utility 1')
ax.set_ylabel('Utility 2')
ax.set_zlabel('Reliability')
ax.set_title('Reliability >= .99', y = 1.0, pad = 0, fontsize = 8)



# 
ax = fig.add_subplot(235, projection='3d')
ax.scatter(df2['Utility1'],
           df2['Utility2'],
           df2['Reliability'],
           color = '#D8D8D8', alpha = 0.1)
ax.scatter(df_sol.loc['u', '3 actor'][0], df_sol.loc['u', '3 actor'][1], df_sol.loc['u', '3 actor'][2], 
        color='r', marker='o', alpha=1.0, label = 'Utility')
ax.scatter(df_sol.loc['ls', '3 actor'][0], df_sol.loc['ls', '3 actor'][1], df_sol.loc['ls', '3 actor'][2], 
        color='y', marker='o', alpha=1.0, label = 'Least Squares')
ax.scatter(df_sol.loc['mm', '3 actor'][0], df_sol.loc['mm', '3 actor'][1], df_sol.loc['mm', '3 actor'][2], 
        color='g', marker='o', alpha=1.0, label = 'MiniMax')
ax.scatter(df_sol.loc['pi', '3 actor'][0], df_sol.loc['pi', '3 actor'][1], df_sol.loc['pi', '3 actor'][2], 
        color='b', marker='o', alpha=1.0, label = 'Power Index')
ax.scatter(df_sol.loc['fb', '3 actor'][0], df_sol.loc['fb', '3 actor'][1], df_sol.loc['fb', '3 actor'][2], 
        color='c', marker='o', alpha=1.0, label = 'Fallback Bargaining')
ax.set_xlabel('Utility 1')
ax.set_ylabel('Utility 2')
ax.set_zlabel('Reliability')
ax.set_title('Reliability as 3rd actor', y = 1.0, pad = 0, fontsize = 8)

# fig.text(0.5, 0.015, 'Reliability', va='center', ha='center')
# fig.text(0.01, 0.5, 'Utility1', va='center', ha='center', rotation='vertical')
# fig.text(0.02, 0.5, 'Utility 2', va='center', ha='center', rotation='vertical')
lines, labels = ax.get_legend_handles_labels()
fig.legend(lines, labels, loc = (0.7, 0.6))
fig.tight_layout()
plt.show()
