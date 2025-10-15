# src/validate.py
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes  # Importar load_diabetes
import sys
import os

# Parámetro de umbral
THRESHOLD = 5000.0  # Ajusta este umbral según el MSE esperado para load_diabetes

# --- Cargar el MISMO dataset que en train.py ---
print("--- Debug: Cargando dataset load_diabetes ---")
X, y = load_diabetes(return_X_y=True, as_frame=True)  # Usar as_frame=True si quieres DataFrames

# División de datos (usar los mismos datos que en entrenamiento no es ideal para validación real,
# pero necesario aquí para que las dimensiones coincidan. Idealmente, tendrías un split dedicado
# o usarías el X_test guardado del entrenamiento si fuera posible)
# Para este ejemplo, simplemente re-dividimos para obtener un X_test con 10 features.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Añadir random_state para consistencia si es necesario
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")  # Debería ser (n_samples, 10)

# --- Cargar modelo previamente entrenado ---
from pathlib import Path
import glob

# --- Buscar el modelo más reciente dentro de mlruns ---
mlruns_dir = Path(os.getcwd()) / "mlruns"

print(f"--- Debug: Buscando modelo dentro de: {mlruns_dir} ---")

# Buscar recursivamente todos los archivos model.pkl en subdirectorios
model_files = list(mlruns_dir.rglob("model.pkl"))

if not model_files:
    print(f"--- ERROR: No se encontró ningún archivo model.pkl dentro de {mlruns_dir} ---")
    print(f"--- Archivos detectados: {list(mlruns_dir.rglob('*'))[:10]} ---")
    sys.exit(1)

# Tomar el más reciente por fecha de modificación
latest_model_path = max(model_files, key=os.path.getmtime)
print(f"--- Debug: Modelo más reciente encontrado en: {latest_model_path} ---")

# Cargar el modelo
try:
    model = joblib.load(latest_model_path)
except Exception as e:
    print(f"--- ERROR al cargar el modelo: {e} ---")
    sys.exit(1)
