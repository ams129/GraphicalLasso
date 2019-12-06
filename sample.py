import numpy as np
import scipy as sp
import seaborn as sns
import GraphicalLasso

data = np.loadtxt("testdata.txt")
data = sp.stats.zscore(data, axis=0)

rho = 0.6
Theta = GraphicalLasso.solve(data, rho)
np.set_printoptions(precision=3,suppress=True)
print(Theta)
