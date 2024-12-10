import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

meses_a_numeros = {
    'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Cargar datos
data = pd.read_csv(r"C:\Users\aialvarado\OneDrive - Agrosuper\Escritorio\datos_historicos_desviaciones.csv")
data['month'] = data['month'].str.lower().map(meses_a_numeros)
print(data.columns)

data['Fecha'] = pd.to_datetime(data[['year', 'month','day']])
data = pd.get_dummies(data,columns=['Desviacion'], drop_first=True)

# Dividir en características y variable objetivo
X = data.drop(columns=['Recuento de Desviación', 'Fecha'])
y = data['Recuento de Desviación']

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuste del modelo con RandomForestRegressor y GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

# Modelo óptimo
best_model = grid_search.best_estimator_

# Predicciones
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Métricas de error
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Entrenamiento - MSE: {mse_train}, MAE: {mae_train}, R²: {r2_train}")
print(f"Prueba - MSE: {mse_test}, MAE: {mae_test}, R²: {r2_test}")

# Intervalos de confianza para las predicciones
n_simulations = 100
predictions = []

for i in range(n_simulations):
    X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
    best_model.fit(X_resampled, y_resampled)
    predictions.append(best_model.predict(X_test))

predictions = np.array(predictions)
mean_prediction = predictions.mean(axis=0)
lower_bound = np.percentile(predictions, 2.5, axis=0)
upper_bound = np.percentile(predictions, 97.5, axis=0)

# Crear tabla con valores predichos
results = pd.DataFrame({
    'Fecha': data['Fecha'][len(y_train):].values,
    'Recuento Real': y_test.values,
    'Predicción': mean_prediction,
    'Intervalo Inferior': lower_bound,
    'Intervalo Superior': upper_bound
})
results['Desviación'] = np.argmax(X_test.values[:, -4:], axis=1) + 1  # Mapear Desviaciones (D1-D4)

# Mostrar tabla
print(results)

# Guardar tabla en archivo
results.to_csv('predicciones_desviaciones.csv', index=False)

# Graficar por tipo de desviación
for desviacion in range(1, 5):
    df_desviacion = results[results['Desviación'] == desviacion]
    plt.figure(figsize=(12, 6))
    plt.plot(df_desviacion['Fecha'], df_desviacion['Recuento Real'], label='Datos Reales')
    plt.plot(df_desviacion['Fecha'], df_desviacion['Predicción'], label='Predicción', color='red')
    plt.fill_between(df_desviacion['Fecha'], df_desviacion['Intervalo Inferior'], 
                     df_desviacion['Intervalo Superior'], color='red', alpha=0.2, label='Intervalo de Confianza')
    plt.title(f"Desviación D{desviacion} - Predicción vs Datos Reales")
    plt.xlabel("Fecha")
    plt.ylabel("Recuento de Desviación")
    plt.legend()
    plt.grid()
    plt.show()