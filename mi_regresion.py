"""
Modelo de Aprendizaje Automático para la Predicción del Precio de
Vivienda en Colombia Mediante Regresión Múltiple y Random Forest

Autores: Santiago Chaparro Riaño, David Chicino Zapata, Deider Luis Basilio Pérez
Institución: Tecnológica del Oriente
Materia: Inteligencia Artificial Avanzada
Docente: José Fabián Díaz Silva
Fecha: 9 de mayo de 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# 1. GENERACIÓN DEL DATASET SIMULADO (100 registros)
#    Variables calibradas con rangos del mercado colombiano
#    (DANE, 2025; Camacol, 2024)
# =============================================================================

np.random.seed(42)
n = 100

ciudades = ['Bogotá', 'Medellín', 'Cali', 'Neiva']
ciudad_col = np.random.choice(ciudades, size=n, p=[0.40, 0.30, 0.20, 0.10])

estrato = np.random.randint(1, 7, size=n)          # 1 a 6
area_m2 = np.random.randint(35, 200, size=n)       # 35 – 200 m²
habitaciones = np.random.randint(1, 6, size=n)     # 1 – 5 hab.
banos = np.random.randint(1, 4, size=n)            # 1 – 3 baños
antiguedad = np.random.randint(0, 31, size=n)      # 0 – 30 años
parqueadero = np.random.randint(0, 2, size=n)      # 0 = no, 1 = sí
ascensor = np.random.randint(0, 2, size=n)         # 0 = no, 1 = sí

# Precio simulado con relación realista a las variables
precio_millones = (
    1.8 * area_m2
    + 25 * estrato
    + 15 * habitaciones
    + 12 * banos
    - 1.5 * antiguedad
    + 20 * parqueadero
    + 15 * ascensor
    + np.where(ciudad_col == 'Bogotá', 50, 0)
    + np.where(ciudad_col == 'Medellín', 30, 0)
    + np.where(ciudad_col == 'Cali', 10, 0)
    + np.random.normal(0, 20, size=n)   # ruido estocástico
)
precio_millones = np.clip(precio_millones, 80, 900)  # límites de mercado

df = pd.DataFrame({
    'area_m2': area_m2,
    'habitaciones': habitaciones,
    'banos': banos,
    'estrato': estrato,
    'ciudad': ciudad_col,
    'antiguedad': antiguedad,
    'parqueadero': parqueadero,
    'ascensor': ascensor,
    'precio_millones': np.round(precio_millones, 1)
})

print("=" * 65)
print("DATASET SIMULADO — Mercado Inmobiliario Colombia")
print("=" * 65)
print(df.head(8).to_string(index=False))
print(f"\nShape: {df.shape}")
print(f"\nEstadísticas descriptivas:\n{df.describe().round(2)}")

# =============================================================================
# 2. PROCESAMIENTO DE DATOS
#    (a) Verificación de nulos
#    (b) One-Hot Encoding para variable categórica 'ciudad'
#    (c) División 70 % entrenamiento / 15 % validación / 15 % prueba
# =============================================================================

print("\n--- Valores nulos por columna ---")
print(df.isnull().sum())

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['ciudad'], drop_first=False)

X = df_encoded.drop('precio_millones', axis=1)
y = df_encoded['precio_millones']

# División en tres subconjuntos
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"\nTamaño de conjuntos:")
print(f"  Entrenamiento : {len(X_train)} registros (70 %)")
print(f"  Validación    : {len(X_val)} registros  (15 %)")
print(f"  Prueba        : {len(X_test)} registros  (15 %)")

# =============================================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
#    - Regresión Lineal Múltiple (modelo base)
#    - Random Forest Regressor   (modelo de ensamble)
# =============================================================================

modelos = {
    "Regresión Lineal Múltiple": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    )
}

resultados = {}

print("\n" + "=" * 65)
print("MÉTRICAS EN CONJUNTO DE PRUEBA")
print("=" * 65)

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    pred_test = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2   = r2_score(y_test, pred_test)

    resultados[nombre] = {
        'modelo': modelo,
        'pred_test': pred_test,
        'mae': mae, 'rmse': rmse, 'r2': r2
    }

    print(f"\n{nombre}")
    print(f"  MAE  : {mae:.2f}  millones COP")
    print(f"  RMSE : {rmse:.2f} millones COP")
    print(f"  R²   : {r2:.4f}")

# =============================================================================
# 4. VERIFICACIÓN DE SOBREAJUSTE (entrenamiento vs. prueba)
# =============================================================================

print("\n" + "=" * 65)
print("VERIFICACIÓN DE SOBREAJUSTE (R² entrenamiento vs. prueba)")
print("=" * 65)

for nombre, res in resultados.items():
    r2_train = r2_score(y_train, res['modelo'].predict(X_train))
    r2_test  = res['r2']
    print(f"{nombre}")
    print(f"  R² entrenamiento : {r2_train:.4f}")
    print(f"  R² prueba        : {r2_test:.4f}")
    print(f"  Diferencia       : {abs(r2_train - r2_test):.4f}")

# =============================================================================
# 5. IMPORTANCIA DE VARIABLES (Random Forest)
# =============================================================================

rf = resultados["Random Forest Regressor"]['modelo']
importancias = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\n--- Importancia de variables (Random Forest) ---")
print(importancias.round(4).to_string())

# =============================================================================
# 6. PREDICCIÓN DE EJEMPLO PRÁCTICO
#    Apartamento: 70 m², 3 hab., 2 baños, estrato 3, Cali,
#    5 años antigüedad, sin parqueadero, sin ascensor
# =============================================================================

ejemplo = pd.DataFrame([{
    'area_m2': 70, 'habitaciones': 3, 'banos': 2,
    'estrato': 3, 'antiguedad': 5,
    'parqueadero': 0, 'ascensor': 0,
    'ciudad_Bogotá': 0, 'ciudad_Cali': 1,
    'ciudad_Medellín': 0, 'ciudad_Neiva': 0
}])
# Asegurar columnas coincidan con el orden del entrenamiento
ejemplo = ejemplo.reindex(columns=X.columns, fill_value=0)

print("\n" + "=" * 65)
print("PREDICCIÓN EJEMPLO: Apto 70 m², estrato 3, Cali, 5 años")
print("=" * 65)
for nombre, res in resultados.items():
    pred = res['modelo'].predict(ejemplo)[0]
    print(f"  {nombre}: ${pred:.1f} millones COP")

# =============================================================================
# 7. GRÁFICAS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Mercado Inmobiliario en Colombia — Comparación de Modelos",
    fontsize=14, fontweight='bold'
)

colores = {
    "Regresión Lineal Múltiple": ("#2196F3", "#0D47A1"),
    "Random Forest Regressor":   ("#4CAF50", "#1B5E20"),
}

# --- 7a. Gráfica: Real vs. Predicho (ambos modelos) ---
for i, (nombre, res) in enumerate(resultados.items()):
    ax = axes[i]
    color_p, color_d = colores[nombre]
    ax.scatter(y_test, res['pred_test'], alpha=0.7, color=color_p,
               edgecolors='white', linewidth=0.5, label='Predicciones')
    lim = [min(y_test.min(), res['pred_test'].min()) - 10,
           max(y_test.max(), res['pred_test'].max()) + 10]
    ax.plot(lim, lim, '--', color=color_d, linewidth=1.5, label='Predicción perfecta')
    ax.set_xlabel('Precio Real (millones COP)', fontsize=10)
    ax.set_ylabel('Precio Predicho (millones COP)', fontsize=10)
    ax.set_title(f'{nombre}\nR² = {res["r2"]:.4f}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# --- 7b. Importancia de variables (Random Forest) ---
ax3 = axes[2]
top10 = importancias.head(10)
bars = ax3.barh(top10.index[::-1], top10.values[::-1],
                color='#4CAF50', edgecolor='white')
ax3.set_xlabel('Importancia relativa', fontsize=10)
ax3.set_title('Importancia de Variables\n(Random Forest)', fontsize=11)
ax3.grid(True, axis='x', alpha=0.3)
for bar, val in zip(bars, top10.values[::-1]):
    ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('resultados_modelos.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGráfica guardada como 'resultados_modelos.png'")

# --- 7c. Comparación de métricas ---
fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
fig2.suptitle("Comparación de Métricas entre Modelos", fontsize=13, fontweight='bold')
nombres = list(resultados.keys())
abr = ["Reg. Lineal\nMúltiple", "Random\nForest"]
metricas_vals = {
    'MAE\n(millones COP)':  [resultados[n]['mae']  for n in nombres],
    'RMSE\n(millones COP)': [resultados[n]['rmse'] for n in nombres],
    'R²':                   [resultados[n]['r2']   for n in nombres],
}
palette = ['#2196F3', '#4CAF50']
for ax, (metrica, vals) in zip(axes2, metricas_vals.items()):
    bars = ax.bar(abr, vals, color=palette, edgecolor='white', width=0.5)
    ax.set_title(metrica, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.03,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparacion_metricas.png', dpi=150, bbox_inches='tight')
plt.show()
print("Gráfica guardada como 'comparacion_metricas.png'")