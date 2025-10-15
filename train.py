import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback
import pathlib

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
# Usar rutas absolutas dentro del workspace del runner
workspace_dir = os.getcwd() # Debería ser /home/runner/work/mlflow-deploy/mlflow-deploy
mlruns_dir = os.path.join(workspace_dir, "mlruns")
##jc tracking_uri = "file://" + os.path.abspath(mlruns_dir)
tracking_uri = pathlib.Path(os.getcwd()) / "mlruns"
# Definir explícitamente la ubicación base deseada para los artefactos
##jc artifact_location = "file://" + os.path.abspath(mlruns_dir)
artifact_location = f"file:///{mlruns_dir.replace(os.sep, '/')}"

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# --- Asegurar que el directorio MLRuns exista ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
##jcc mlflow.set_tracking_uri(tracking_uri)
tracking_uri = f"file:///{mlruns_dir.replace(os.sep, '/')}"

# --- Crear o Establecer Experimento Explícitamente con Artifact Location ---
experiment_name = "CI-CD-Lab2"
experiment_id = None # Inicializar variable
try:
    # Intentar crear el experimento, proporcionando la ubicación del artefacto
    ##jcc experiment_id = mlflow.create_experiment(
    ##jcc    name=experiment_name,
    ##jcc    artifact_location=artifact_location # ¡Forzar la ubicación aquí!
    ##jcc)
    experiment_id = mlflow.create_experiment("Experiment_Local", artifact_location=str(mlruns_dir))
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---")
        # Obtener el experimento existente para conseguir su ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---")
            # Opcional: Verificar si la ubicación del artefacto es la correcta
            if experiment.artifact_location != artifact_location:
                 print(f"--- WARNING: La ubicación del artefacto del experimento existente ('{experiment.artifact_location}') NO coincide con la deseada ('{artifact_location}')! ---")
        else:
            # Esto no debería ocurrir si RESOURCE_ALREADY_EXISTS fue el error
            print(f"--- ERROR: No se pudo obtener el experimento existente '{experiment_name}' por nombre. ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e # Relanzar otros errores

# Asegurarse de que tenemos un experiment_id válido
if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}'. ---")
    sys.exit(1)

# --- Cargar Datos y Entrenar Modelo ---
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

    # --- Iniciar Run de MLflow ---
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---")
run = None
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        # Forzar tracking URI con formato correcto
        mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns').replace(os.sep, '/')}")

        mlflow.log_metric("mse", mse)
        print(f"--- Debug: Intentando log_model con artifact_path='model' ---")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="linear_regression_model"
        )
        print(f"✅ Modelo registrado correctamente. MSE: {mse:.4f}")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID intentado: {experiment_id}")
    if run:
         print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}")
    else:
         print("El objeto Run no se creó con éxito.")
    sys.exit(1)
