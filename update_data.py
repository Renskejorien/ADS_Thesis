# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import data
df = pd.read_csv('City2.csv', sep=',')
df = df[df.columns[1::]]
df1 = df[df.columns[0:6]].abs()
df2 = df[df.columns[6::]]
df = df1.join(df2)
print(df[df.columns[0:6]].head())
print(df[df.columns[6:10]].head())
df.to_csv('City2.csv', sep=',')
