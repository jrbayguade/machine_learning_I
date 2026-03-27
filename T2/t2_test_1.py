import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# seed for reproducibility
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)
np.var(data)

plt.hist(data, bins=30, color='skyblue')
plt.show()

print("Mean:", np.mean(data))
print("Variance:", np.var(data))
print("Skewness:", skew(data))
print("Kurtosis:", kurtosis(data))