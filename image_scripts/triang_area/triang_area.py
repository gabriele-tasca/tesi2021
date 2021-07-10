import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import fract

df = pd.read_csv("stats.csv", sep=";")

ratios = df["Area Ratio"]


# df[ df["Nome"] == "Adamello"]
np.min(ratios)

plt.hist(ratios, bins=20)
plt.xlabel("Area Ratio")

plt.savefig("area-ratio-hist.png", bbox_inches='tight')

# np.mean(ratios)
# np.std(ratios)