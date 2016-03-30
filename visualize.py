import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import andrews_curves, parallel_coordinates

data = pd.read_csv('train.csv')

plt.figure()
parallel_coordinates(data, 'TARGET')
plt.show()
