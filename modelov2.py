import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Mapeo de meses
meses_a_numeros = {
    'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

# Cargar datos
data = pd.read_csv(r"C:\Users\aialvarado\OneDrive - Agrosuper\Escritorio\datos_historicos_desviaciones.csv")
data['month'] = data['month'].str.lower().map(meses_a_numeros)
data['Fecha'] = pd.to_datetime(data[['year', 'month', 'day']])
data = pd.get_dummies(data, columns=['Desviacion'], drop_first=True)

# Características y variable objetivo
X = data.drop(columns=['Recuento de Desviación', 'Fecha'])
y = data['Recuento de Desviación']

# División en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuste del modelo con GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predicción extendida hasta diciembre
n_predicciones = (12 - X['month'].max()) * len(X_test) // 12
X_test_extended = X_test.copy()
for i in range(n_predicciones):
    X_test_extended = pd.concat([X_test_extended, X_test.iloc[[i % len(X_test)]]], ignore_index=True)

# Intervalos de confianza
n_simulations = 100
predictions = []
for i in range(n_simulations):
    X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
    best_model.fit(X_resampled, y_resampled)
    predictions.append(best_model.predict(X_test_extended))

predictions = np.array(predictions)
mean_prediction = predictions.mean(axis=0)
lower_bound = np.percentile(predictions, 2.5, axis=0)
upper_bound = np.percentile(predictions, 97.5, axis=0)

# Tabla de resultados
results = pd.DataFrame({
    'Fecha': pd.date_range(start=data['Fecha'].max(), periods=len(mean_prediction)),
    'Predicción': mean_prediction,
    'Intervalo Inferior': lower_bound,
    'Intervalo Superior': upper_bound
})
results['Desviación'] = np.argmax(X_test_extended.values[:, -4:], axis=1) + 1  # Mapear desviaciones (D1-D4)

# Gráficos
def graficar_por_desviacion(results, general=False):
    if general:
        plt.figure(figsize=(14, 7))
        for desviacion in range(1, 5):
            df_desviacion = results[results['Desviación'] == desviacion]
            plt.plot(df_desviacion['Fecha'], df_desviacion['Predicción'], label=f'Desviación D{desviacion}')
            plt.fill_between(
                df_desviacion['Fecha'], df_desviacion['Intervalo Inferior'], df_desviacion['Intervalo Superior'],
                alpha=0.2
            )
        plt.title("Gráfico General - Predicción por Desviación")
        plt.xlabel("Fecha")
        plt.ylabel("Recuento de Desviación")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        for desviacion in range(1, 5):
            df_desviacion = results[results['Desviación'] == desviacion]
            plt.figure(figsize=(12, 6))
            plt.plot(df_desviacion['Fecha'], df_desviacion['Predicción'], label='Predicción', color='blue')
            plt.fill_between(
                df_desviacion['Fecha'], df_desviacion['Intervalo Inferior'], df_desviacion['Intervalo Superior'],
                color='blue', alpha=0.2, label='Intervalo de Confianza'
            )
            plt.title(f"Desviación D{desviacion} - Predicción Extendida")
            plt.xlabel("Fecha")
            plt.ylabel("Recuento de Desviación")
            plt.legend()
            plt.grid()
            plt.show()

# Llamada a la función
opcion = input("Selecciona el gráfico a mostrar (separado/general): ").strip().lower()
if opcion == "general":
    graficar_por_desviacion(results, general=True)
elif opcion == "separado":
    graficar_por_desviacion(results, general=False)
else:
    print("Opción no válida. Introduce 'separado' o 'general'.")