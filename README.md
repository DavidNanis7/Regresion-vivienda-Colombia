# Modelo de Aprendizaje Automático para la Predicción del Precio de Vivienda en Colombia

Proyecto académico desarrollado para la asignatura **Inteligencia Artificial Avanzada** de la **Tecnológica del Oriente**.

Este repositorio contiene un modelo de aprendizaje automático supervisado para estimar el precio de vivienda en Colombia a partir de variables físicas, geográficas y socioeconómicas del inmueble. El trabajo compara un modelo base de **Regresión Lineal Múltiple** con un modelo de ensamble más robusto: **Random Forest Regressor**.

---

## Integrantes

- Santiago Chaparro Riaño
- David Chicino Zapata
- Deider Luis Basilio Pérez

**Docente:** José Fabián Díaz Silva  
**Materia:** Inteligencia Artificial Avanzada  
**Fecha:** 9 de mayo de 2026

---

## Descripción del proyecto

El mercado inmobiliario colombiano presenta variaciones importantes en el precio de la vivienda según factores como la ciudad, el área, el estrato socioeconómico, la antigüedad del inmueble y características adicionales como parqueadero o ascensor.

El objetivo de este proyecto es construir y evaluar un modelo de inteligencia artificial capaz de predecir el precio aproximado de una vivienda en millones de pesos colombianos, utilizando técnicas de aprendizaje automático supervisado.

El proyecto busca responder la siguiente pregunta:

> ¿Es posible construir un modelo de aprendizaje automático que prediga con precisión el precio de un apartamento en Colombia a partir de sus características físicas y socioeconómicas?

---

## Tipo de problema

Este proyecto corresponde a un problema de **regresión supervisada**.

- Es **supervisado** porque el modelo aprende a partir de ejemplos que ya tienen una respuesta conocida: el precio de la vivienda.
- Es de **regresión** porque la variable objetivo es numérica y continua: el precio estimado en millones de pesos colombianos.

---

## Objetivo general

Diseñar, entrenar y evaluar un modelo de aprendizaje automático para predecir el precio de vivienda en Colombia mediante la comparación entre Regresión Lineal Múltiple y Random Forest Regressor.

---

## Objetivos específicos

- Construir un dataset académico simulado con variables representativas del mercado inmobiliario colombiano.
- Procesar los datos mediante verificación de nulos, codificación de variables categóricas y separación de variables predictoras y objetivo.
- Dividir los datos en conjuntos de entrenamiento, validación y prueba.
- Entrenar un modelo base de Regresión Lineal Múltiple.
- Entrenar un modelo avanzado de Random Forest Regressor.
- Evaluar el rendimiento con métricas de regresión: MAE, RMSE y R².
- Comparar el desempeño de ambos modelos.
- Analizar posibles sesgos, limitaciones y consideraciones éticas del uso de IA en el mercado inmobiliario.

---

## Dataset utilizado

El proyecto utiliza un conjunto de datos simulado de **100 registros**, creado con fines académicos. Los valores fueron generados con rangos realistas inspirados en el mercado inmobiliario colombiano.

### Variables del dataset

| Variable | Tipo | Descripción |
|---|---|---|
| `area_m2` | Numérica | Área total del inmueble en metros cuadrados |
| `habitaciones` | Numérica | Número de habitaciones |
| `banos` | Numérica | Número de baños |
| `estrato` | Numérica | Estrato socioeconómico del inmueble, entre 1 y 6 |
| `ciudad` | Categórica | Ciudad donde se ubica el inmueble: Bogotá, Medellín, Cali o Neiva |
| `antiguedad` | Numérica | Años de construcción del inmueble |
| `parqueadero` | Binaria | 1 si tiene parqueadero, 0 si no tiene |
| `ascensor` | Binaria | 1 si tiene ascensor, 0 si no tiene |
| `precio_millones` | Objetivo | Precio de la vivienda en millones de pesos colombianos |

---

## Modelos implementados

### 1. Regresión Lineal Múltiple

Se utiliza como modelo base. Permite observar una relación directa entre las variables de entrada y el precio estimado de la vivienda.

Este modelo es útil porque es fácil de interpretar y permite tener una primera aproximación al problema.

### 2. Random Forest Regressor

Se utiliza como modelo avanzado. Este modelo combina múltiples árboles de decisión para mejorar la capacidad predictiva y capturar relaciones no lineales entre las variables.

En este proyecto se configuró con:

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
```

- `n_estimators=100`: indica que el bosque está compuesto por 100 árboles.
- `max_depth=5`: limita la profundidad de los árboles para reducir el riesgo de sobreajuste.
- `random_state=42`: permite reproducir los mismos resultados en cada ejecución.

---

## Flujo general del proyecto

```text
1. Generación del dataset simulado
2. Creación del DataFrame con Pandas
3. Verificación de valores nulos
4. Codificación de la variable ciudad con One-Hot Encoding
5. Separación de variables predictoras X y variable objetivo y
6. División de datos en entrenamiento, validación y prueba
7. Entrenamiento de Regresión Lineal Múltiple
8. Entrenamiento de Random Forest Regressor
9. Evaluación con MAE, RMSE y R²
10. Comparación de modelos
11. Análisis de importancia de variables
12. Predicción de un caso nuevo
13. Generación de gráficas de resultados
```

---

## División de datos

El dataset se divide de la siguiente manera:

| Conjunto | Porcentaje | Uso |
|---|---:|---|
| Entrenamiento | 70 % | El modelo aprende los patrones de los datos |
| Validación | 15 % | Permite revisar el comportamiento del modelo antes de la prueba final |
| Prueba | 15 % | Evalúa el rendimiento final sobre datos no vistos |

---

## Métricas de evaluación

Para evaluar los modelos se utilizan métricas propias de problemas de regresión:

| Métrica | Significado |
|---|---|
| MAE | Error absoluto medio. Indica cuánto se equivoca el modelo en promedio |
| RMSE | Raíz del error cuadrático medio. Penaliza más los errores grandes |
| R² | Coeficiente de determinación. Indica qué porcentaje de la variación del precio explica el modelo |

---

## Resultados esperados del trabajo

En el informe académico se presentan los siguientes resultados comparativos:

| Modelo | MAE | RMSE | R² | Interpretación |
|---|---:|---:|---:|---|
| Regresión Lineal Múltiple | 35.2 | 42.8 | 0.82 | Buen desempeño general |
| Random Forest Regressor | 22.5 | 30.1 | 0.91 | Mayor capacidad predictiva |

El modelo **Random Forest Regressor** obtiene el mejor desempeño, ya que presenta menor error y mayor capacidad explicativa sobre el precio de vivienda.

---

## Predicción de ejemplo

El proyecto incluye una predicción práctica para un apartamento con las siguientes características:

| Característica | Valor |
|---|---|
| Área | 70 m² |
| Habitaciones | 3 |
| Baños | 2 |
| Estrato | 3 |
| Ciudad | Cali |
| Antigüedad | 5 años |
| Parqueadero | No |
| Ascensor | No |

Resultado esperado según el informe:

- Regresión Lineal Múltiple: aproximadamente **289 millones COP**.
- Random Forest Regressor: aproximadamente **305 millones COP**.

---

## Gráficas generadas

Al ejecutar el archivo principal, el programa genera y guarda las siguientes imágenes:

| Archivo | Descripción |
|---|---|
| `resultados_modelos.png` | Gráfica de precio real vs. precio predicho e importancia de variables |
| `comparacion_metricas.png` | Comparación visual de MAE, RMSE y R² entre modelos |

Estas gráficas permiten observar de forma visual qué tan cerca están las predicciones del precio real y cuáles variables influyen más en el modelo Random Forest.

---

## Estructura sugerida del repositorio

```text
Regresion-vivienda-Colombia/
│
├── mi_regresion.py
├── README.md
├── Creacion_modelo_IA_avanzada_APA7.docx
├── resultados_modelos.png
├── comparacion_metricas.png
└── requirements.txt
```

> Nota: si el archivo `requirements.txt` no existe, se puede crear con las dependencias indicadas en la sección de instalación.

---

## Tecnologías utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/DavidNanis7/Regresion-vivienda-Colombia.git
cd Regresion-vivienda-Colombia
```

### 2. Crear un entorno virtual, opcional pero recomendado

```bash
python -m venv venv
```

Activar entorno en Windows:

```bash
venv\Scripts\activate
```

Activar entorno en Linux o macOS:

```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install numpy pandas matplotlib scikit-learn
```

También puedes crear un archivo `requirements.txt` con este contenido:

```txt
numpy
pandas
matplotlib
scikit-learn
```

Y luego ejecutar:

```bash
pip install -r requirements.txt
```

### 4. Ejecutar el proyecto

```bash
python mi_regresion.py
```

---

## Qué hace el archivo `mi_regresion.py`

El archivo principal realiza todo el flujo del proyecto:

1. Genera un dataset simulado de 100 registros.
2. Crea una tabla de datos con Pandas.
3. Verifica si existen valores nulos.
4. Convierte la variable `ciudad` en columnas numéricas mediante One-Hot Encoding.
5. Divide los datos en entrenamiento, validación y prueba.
6. Entrena los modelos Regresión Lineal Múltiple y Random Forest Regressor.
7. Calcula las métricas MAE, RMSE y R².
8. Verifica señales de sobreajuste comparando R² de entrenamiento y prueba.
9. Calcula la importancia de variables del Random Forest.
10. Realiza una predicción para un apartamento nuevo.
11. Genera y guarda gráficas de resultados.

---

## Consideraciones éticas

El modelo no debe utilizarse como avalúo oficial ni como única fuente para tomar decisiones de compra, venta o financiación de vivienda.

Algunos riesgos identificados son:

| Riesgo | Posible impacto | Mitigación |
|---|---|---|
| Sesgo por ciudad | Mejor rendimiento en ciudades con más datos | Balancear el dataset por ciudad |
| Sesgo por estrato | Predicciones injustas para ciertos grupos socioeconómicos | Evaluar métricas por estrato |
| Uso como avalúo oficial | Decisiones económicas equivocadas | Informar que es una estimación académica |
| Datos personales | Riesgo de privacidad si se usan datos reales | Anonimizar la información |

---

## Limitaciones

- El dataset es simulado y tiene fines académicos.
- Los resultados no deben extrapolarse directamente al mercado real.
- No se incluyen variables de localización intraurbana como barrio, cercanía a vías, transporte o servicios.
- La muestra de 100 registros es pequeña para una aplicación comercial.
- No se modela la evolución temporal del mercado inmobiliario.

---

## Posibles mejoras futuras

- Usar datos reales de portales inmobiliarios, DANE, Camacol o IGAC.
- Incorporar variables como barrio, tipo de inmueble, cercanía a transporte y zonas comerciales.
- Aplicar búsqueda de hiperparámetros con `GridSearchCV` o `RandomizedSearchCV`.
- Comparar modelos adicionales como Gradient Boosting, XGBoost o redes neuronales.
- Crear una interfaz web para consultar predicciones.
- Evaluar equidad por ciudad, estrato y tipo de vivienda.

---

## Documentación académica

El informe completo del proyecto se encuentra en el archivo:

```text
Creacion_modelo_IA_avanzada_APA7.docx
```

Este documento contiene:

- Resumen
- Introducción
- Identificación y justificación del problema
- Revisión de literatura
- Diseño del modelo y metodología
- Resultados de entrenamiento y evaluación
- Discusión sobre equidad y ética
- Limitaciones
- Conclusiones y recomendaciones
- Referencias
- Anexo del repositorio

---

## Nota académica

Este proyecto fue desarrollado con fines educativos para demostrar el proceso completo de creación de un modelo de inteligencia artificial avanzada aplicado a un problema realista: la predicción del precio de vivienda en Colombia.

El modelo funciona como una herramienta de apoyo y aprendizaje, no como un sistema profesional de avalúo inmobiliario.
