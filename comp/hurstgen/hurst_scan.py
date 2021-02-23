from hurst import compute_Hc
import numpy as np
import matplotlib.pyplot as plt

series = np.genfromtxt('frac1d.csv',delimiter=' ')

H, c, data = compute_Hc(series, kind='random_walk', min_window=200 ,simplified=False)

print(H,c)

# Plot
f, ax = plt.subplots()
ax.plot(data[0], c*data[0]**H, color="deepskyblue")
ax.scatter(data[0], data[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()
