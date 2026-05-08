import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. DATOS REALISTAS (Apartamentos en Bogotá/Medellín/Cali)
# Área en m2 y Precio en Millones de Pesos (COP)
datos = {
    'area_m2': [45, 50, 62, 75, 80, 95, 110, 125, 140, 155],
    'precio_millones': [180, 210, 260, 310, 330, 400, 460, 520, 580, 640]
}

df = pd.DataFrame(datos)

# 2. PREPARACIÓN
X = df[['area_m2']] # Variable independiente (lo que sabemos)
y = df['precio_millones'] # Variable dependiente (lo que queremos predecir)

# Dividimos para entrenar (80%) y probar (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. ENTRENAMIENTO DEL MODELO
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 4. PRUEBA DE PREDICCIÓN
# Cuánto valdría un apto de 70m2 y uno de 120m2
areas_nuevas = [[70], [120]]
predicciones = modelo.predict(areas_nuevas)

print(f"Precio estimado para 70m2: ${predicciones[0]:.2f} Millones")
print(f"Precio estimado para 120m2: ${predicciones[1]:.2f} Millones")

# 5. GRÁFICA PARA EL TRABAJO 
plt.scatter(df['area_m2'], df['precio_millones'], color='blue', label='Ventas reales')
plt.plot(df['area_m2'], modelo.predict(X), color='red', label='Tendencia del mercado')
plt.title('Mercado Inmobiliario en Colombia: Área vs Precio')
plt.xlabel('Área (Metros Cuadrados)')
plt.ylabel('Precio (Millones COP)')
plt.legend()
plt.grid(True)
plt.show()