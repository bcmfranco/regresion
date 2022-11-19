# 1. Encender el entorno virtual:
## source /env/bin/activate
# 2. Correr el script
## python regresion.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


datos = pd.read_csv("datas/ingreso.csv")

## Mostrar el gráfico
# plt.ylabel("Ingreso (S)")
# plt.xlabel("Promedio de horas semanales trabajadas")
# plt.scatter(datos["horas"], datos["ingreso"], color="pink")
# plt.show()

regresion = linear_model.LinearRegression()
horas = datos["horas"].values.reshape((-1, 1))

modelo = regresion.fit(horas, datos["ingreso"])

print("Intersección (b)", modelo.intercept_)
print("Pendiente (m)", modelo.coef_)

entrada = [[39], [40], [43], [44]]
print(modelo.predict(entrada))

plt.scatter(entrada, modelo.predict(entrada), color="red")
plt.plot(entrada, modelo.predict(entrada), color="black")

# Mostrar el gráfico
plt.ylabel("Ingreso (S)")
plt.xlabel("Promedio de horas semanales trabajadas")
plt.scatter(datos["horas"], datos["ingreso"], color="pink")
plt.show()