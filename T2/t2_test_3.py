from sklearn.datasets import load_iris
from imblearn.datasets import make_imbalance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sampling strategy: reduce the number of samples in class 0 to 10, class 1 to 20, and keep class 2 with 47 samples
iris = load_iris(as_frame=False)

sampling_strategy = {0: 10, 1: 20, 2: 47}
X, y = make_imbalance(iris.data, iris.target, 
                      sampling_strategy=sampling_strategy, random_state=42)

print("Tamaño de X:", X.shape)
print("Tamaño de y:", y.shape)

# Se visualiza el conjunto de datos desequilibrado
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Imbalanced Dataset')
plt.show()

# Se muestra la distribución de clases
unique, counts = np.unique(y, return_counts=True)
print("Distribución de clases:")
for cls, count in zip(unique, counts):
    print(f"Clase {cls}: {count} muestras")


autopct = "%.2f"
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
autopct = "%.2f"
pd.Series(iris.target).value_counts().plot.pie(autopct=autopct, ax=axs[0], title="Distribución original")
axs[0].set_title("Distribución original")
pd.Series(y).value_counts().plot.pie(autopct=autopct, ax=axs[1], title="Distribución desequilibrada")
axs[1].set_title("Distribución desequilibrada")
fig.tight_layout()
plt.show()