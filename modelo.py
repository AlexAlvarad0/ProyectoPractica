import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Diccionario para mapear meses en texto a números
meses_a_numeros = {
    'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Leer los datos
data = pd.read_csv(r"C:\Users\aialvarado\OneDrive - Agrosuper\Escritorio\datos_historicos_desviaciones.csv")

# Convertir los nombres de los meses a números
data['month'] = data['month'].str.lower().map(meses_a_numeros)

# Crear la columna 'Fecha' combinando 'year', 'month' y 'day'
data['Fecha'] = pd.to_datetime(data[['year', 'month', 'day']])

# Generar variables dummies para la columna 'Desviacion'
data = pd.get_dummies(data, columns=['Desviacion'], drop_first=True)

# Seleccionar características (X) y variable objetivo (y)
X = data[['month', 'year', 'day'] + [col for col in data.columns if 'Desviacion' in col]]
y = data['Recuento de Desviación']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(data['Fecha'][:len(y_pred_train)], y_train, label='Datos reales (entrenamiento)')
plt.plot(data['Fecha'][:len(y_train)], y_pred_train, '--', label='Predicciones (entrenamiento)', color='green')
plt.plot(data['Fecha'][len(y_train):], y_test, label='Datos reales (prueba)')
plt.plot(data['Fecha'][len(y_train):], y_pred_test, '--', label='Predicciones (prueba)', color='red')
plt.title("Predicción de Desviaciones")
plt.xlabel("Fecha")
plt.ylabel("Recuento de Desviación")
plt.legend()
plt.show()