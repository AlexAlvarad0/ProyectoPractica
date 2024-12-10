import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


## Mapear los meses al formato numérico ##
data = pd.read_csv(r"C:\Users\aialvarado\OneDrive - Agrosuper\Escritorio\datos_historicos_desviaciones.csv")
data['Fecha'] = pd.to_datetime([['year', 'month', 'day']])
print(data.columns)

data = pd.get_dummies(data, columns=['Tipo'], drop_first = True)
meses_espanol={ 'junio': 6, 'julio':7, 'agosto':8, 'septiembre': 9, 'octubre':10}

#data['month'] = data['Fecha'].dt.month
#data['year'] = data['Fecha'].dt.year
#data['day'] = data['Fecha'].dt.day

#print(data.head())

## Dividir datos para entreno ##

#df_d1 = data[data['Desviación']== 'D1']

#df_d1 = df_d1.set_index('date').resample('ME').sum()

#df_d1['month'] = df_d1.index.month

#df_d1['year'] = df_d1.index.year

X = data[['month', 'year', 'day'] + [col for col in data.columns if 'Tipo' in col]]
y = data['Recuento de Desviación']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle = False)

## Entrenar modelo ##

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred_train= model.predict(X_train)
y_pred_test = model.predict(X_test)
#error = mean_squared_error(y_test, y_pred)

#print(f"Error cuadrático medio : {error}")

## Generar predicciones ##

#future_years = [2024] * 12
#future_months = list(range(1,13))

#X_future = pd.DataFrame({'month':future_months, 'year':future_years})

#future_predictions = model.predict(X_future)

#print("Predicciones futuras:", future_predictions)

## Visualizar resultados ##

plt.figure(figsize=(12,6))
plt.plot(data['Fecha'][:len(y_pred_train)],y_train, label= 'Datos reales (entrenamiento)')
plt.plot(data['Fecha'][:len(y_train)],y_pred_train, '--', label = 'Predicciones (entrenamiento)', color = 'green')
plt.plot(data['Fecha'][len(y_train):],y_test, label = 'Datos reales (prueba)')
plt.plot(data['Fecha'][len(y_train):],y_pred_test, '--', label = 'Predicciones (prueba)',color = 'red')
plt.title("Predicción de Desviaciones")
plt.xlabel("Fecha")
plt.ylabel("Recuento de Desviación")
plt.legend()
plt.show() 