# 1. Encender el entorno virtual:
## source /env/bin/activate
# 2. Correr el script
## python kvecinos.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

clientes = pd.read_csv("datas/creditos.csv")

# Separo entre Pagadores y Deudores
pagadores = clientes[clientes["cumplio"]==1]
deudores = clientes[clientes["cumplio"]==0]

# Dibijo el gráfico
# plt.scatter(pagadores["edad"], pagadores["credito"], marker="*", s=150, color="skyblue", label="pagador (clase:1)")
# plt.scatter(deudores["edad"], deudores["credito"], marker="*", s=150, color="red", label="deudor (clase:2)")
# plt.ylabel("Mondo del crédito")
# plt.xlabel("Edad")
# plt.legend(["pagadores", 'deudores'])
# plt.show()

# Escalo los datos
datos = clientes[["edad", "credito"]]
clase = clientes["cumplio"]
escalador = preprocessing.MinMaxScaler()
datos = escalador.fit_transform(datos)

# Creo el clasificador
clasificador = KNeighborsClassifier(n_neighbors=5) # cantidad de vecinos que voy a tomar en cuenta
clasificador.fit(datos, clase)

# Vamos a suponer que un cliente llamado abel de 53 años quiere 350k
abel_edad = 45
abel_monto = 300000

#Escalo los datos del nuevo solicitante
solicitante = escalador.transform([[abel_edad, abel_monto]])
clasificacion = clasificador.predict(solicitante)[0]
print("Clase", clasificacion)
print("Probabilidades por clase", clasificador.predict_proba(solicitante))

# Dibujo el gráfico anterior pero sumándole al solicitante
plt.scatter(pagadores["edad"], pagadores["credito"], marker="*", s=150, color="skyblue", label="pagador (clase:1)")
plt.scatter(deudores["edad"], deudores["credito"], marker="*", s=150, color="red", label="deudor (clase:2)")
plt.scatter(abel_edad, abel_monto, marker="P", s=250, color="green", label="Abel")
plt.ylabel("Mondo del crédito")
plt.xlabel("Edad")
plt.legend(["pagadores", 'deudores'])
plt.show()

