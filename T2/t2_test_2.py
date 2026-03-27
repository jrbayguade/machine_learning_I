import pandas as pd
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
train = pd.DataFrame(iris.data)
test = pd.DataFrame(iris.target)

# Reserva el 20% de los datos de entrenamiento y prueba para la evaluación del modelo
X_train, X_test, y_train, y_test = train_test_split(train, test, 
                                                    test_size=0.2, shuffle=True, random_state=42)

# uso la misma función anterior para el conjunto de validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.25, random_state=42)

print("Tamaño de X_train:", X_train.shape)
print("Tamaño de X_test:", X_test.shape)
print("Tamaño de X_val:", X_val.shape)
print("Tamaño de y_train:", y_train.shape)
print("Tamaño de y_test:", y_test.shape)
print("Tamaño de y_val:", y_val.shape)
