## 🚀 app.py (Base para la Puesta en Producción/Inferencia)

import pickle
import pandas as pd
import numpy as np
import warnings

# Ignorar advertencias de Scikit-learn
warnings.filterwarnings("ignore")

# --- 1. Cargar el Pipeline Entrenado ---
try:
    with open('titanic_pipeline.pkl', 'rb') as file:
        pipeline_modelo = pickle.load(file)
    print("Pipeline de modelo cargado exitosamente.")
except FileNotFoundError:
    print("ERROR: El archivo 'titanic_pipeline.pkl' no fue encontrado. Ejecuta 5_pipeline.ipynb primero.")
    pipeline_modelo = None


# --- 2. Función de Predicción ---

def predecir_supervivencia(data_pasajero: dict) -> str:
    """
    Realiza una predicción de supervivencia para un solo pasajero 
    utilizando el pipeline de Machine Learning cargado.

    Args:
        data_pasajero: Diccionario con las características del pasajero. 
                       Debe contener: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

    Returns:
        Una cadena que indica si se predice que sobrevivió o no.
    """
    if pipeline_modelo is None:
        return "Error: Modelo no disponible."

    # Convertir el diccionario de entrada a DataFrame (formato que espera el pipeline)
    df_prediccion = pd.DataFrame([data_pasajero])
    
    # El Pipeline se encarga automáticamente de:
    # 1. Aplicar las transformaciones (Quantile, MinMax, OneHot)
    # 2. Realizar la predicción con el clasificador (KNN)
    
    prediccion = pipeline_modelo.predict(df_prediccion)[0]
    
    if prediccion == 1:
        return "PREDICCIÓN: Sí sobrevivió."
    else:
        return "PREDICCIÓN: No sobrevivió."


# --- 3. Ejemplo de Uso y Prueba ---

if __name__ == "__main__":
    print("\n--- INICIO DE PRUEBAS DE INFERENCIA ---")
    
    # Pasajero 1: Mujer de 30 años, Primera Clase, Tarifa alta. (Debería Sobrevivir)
    pasajero_1 = {
        'Pclass': 1, 
        'Sex': 'female', 
        'Age': 30.0, 
        'SibSp': 0, 
        'Parch': 0, 
        'Fare': 100.0, 
        'Embarked': 'S'
    }
    
    # Pasajero 2: Hombre de 45 años, Tercera Clase, Tarifa baja. (Debería NO Sobrevivir)
    pasajero_2 = {
        'Pclass': 3, 
        'Sex': 'male', 
        'Age': 45.0, 
        'SibSp': 1, 
        'Parch': 0, 
        'Fare': 15.0, 
        'Embarked': 'Q'
    }
    
    print("\nPasajero 1 (Mujer, 1ra Clase):")
    print(predecir_supervivencia(pasajero_1))
    
    print("\nPasajero 2 (Hombre, 3ra Clase):")
    print(predecir_supervivencia(pasajero_2))