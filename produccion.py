from flask import Flask, request, jsonify
import pickle
import numpy as np

modelo = None

# Cargar el modelo
with open("modelo.pkl", 'rb') as file:
    modelo = pickle.load(file)

app = Flask(__name__)

@app.route('/predecir', methods=['POST'])
def predict():
    # obtener json
    data = request.get_json(force=True)
    
    # convertir los datos a un arreglo de Numpy
    input_data = np.array(data['input']).reshape(1, -1)
    
    # Hacer predicción
    prediccion = modelo.predict(input_data)
    
    # regresar predicción en formato json
    return jsonify({'prediccion': int(prediccion[0])})

if __name__ == '__main__':
    app.run(debug=True)