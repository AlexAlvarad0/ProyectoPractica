import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('AirPassengers.csv')

# Convertir la columna de fecha a datetime
df['Month'] = pd.to_datetime(df['Month'])

# Establecer la fecha como índice
df.set_index('Month', inplace=True)

# Dividir datos en entrenamiento y prueba
train_data = df.iloc[:-12]  # Últimos 12 meses como conjunto de prueba
test_data = df.iloc[-12:]

# Modelo de Suavizado Exponencial Holt-Winters
model = ExponentialSmoothing(
    train_data['#Passengers'], 
    trend='add', 
    seasonal='add', 
    seasonal_periods=12
).fit()

# Realizar predicciones
forecast = model.forecast(12)
forecast_index = pd.date_range(
    start=test_data.index[0], 
    periods=12, 
    freq='M'
)
forecast_series = pd.Series(forecast, index=forecast_index)

# Calcular métricas de error
mae = mean_absolute_error(test_data['#Passengers'], forecast)
mse = mean_squared_error(test_data['#Passengers'], forecast)
rmse = np.sqrt(mse)

print("Métricas de Error:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['#Passengers'], label='Datos de Entrenamiento')
plt.plot(test_data.index, test_data['#Passengers'], label='Datos de Prueba')
plt.plot(forecast_index, forecast, color='red', label='Predicción')
plt.title('Forecasting de Pasajeros Aéreos')
plt.xlabel('Fecha')
plt.ylabel('Número de Pasajeros')
plt.legend()
plt.tight_layout()
plt.savefig('airline_passengers_forecast.png')

# Exportar predicciones a CSV
forecast_df = pd.DataFrame({
    'Fecha': forecast_index,
    'Pasajeros_Predichos': forecast
})
forecast_df.to_csv('airline_passengers_forecast.csv', index=False)