from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load
from werkzeug.utils import secure_filename
import os
import requests
import json
import pandas as pd
from joblib import load, dump

#Cargar el modelo
dt = load('../modelo.joblib')

#Generar el servidor (Back-end)
servidorWeb = Flask(__name__)


@servidorWeb.route("/formulario",methods=['GET'])
def formulario():
    return render_template('pagina.html')

# Predicción del modelo a partir de un formulario
@servidorWeb.route('/modeloForm', methods=['POST'])
def modeloForm():
    #Procesar datos de entrada 
    contenido = request.form
    
    datosEntrada = np.array([
            contenido['Pregnancies'],
            contenido['Glucose'],
            contenido['BloodPressure'],
            contenido['SkinThickness'],
            contenido['Insulin'],
            contenido['BMI'],
            contenido['DiabetesPedigreeFunction'],
            contenido['Age']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

#Envio de datos a través de Archivos
@servidorWeb.route('/modeloFile', methods=['POST'])
def modeloFile():
    f = request.files['file']
    filename=secure_filename(f.filename)
    path=os.path.join(os.getcwd(),'files',filename)
    f.save(path)
    file = open(path, "r")
    
    for x in file:
        info=x.split()
    print(info)
    datosEntrada = np.array([
            float(info[0]),
            float(info[1]),
            float(info[2]),
            float(info[3]),
            float(info[4]),
            float(info[5]),
            float(info[6]),
            float(info[7])
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

#Envio de datos a través de JSON
@servidorWeb.route('/modelo', methods=['POST'])
def modelo():
    #Procesar datos de entrada 
    contenido = request.json
    print(contenido)
    datosEntrada = np.array([
            contenido['Pregnancies'],
            contenido['Glucose'],
            contenido['BloodPressure'],
            contenido['SkinThickness'],
            contenido['Insulin'],
            contenido['BMI'],
            contenido['DiabetesPedigreeFunction'],
            contenido['Age']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

# Reentrenar modelo
@servidorWeb.route('/reEntrenar', methods=['POST'])
def reEntrenar():
    url = 'http://localhost:8080/Base/consultarRegistros'
    r = requests.get(url = url)
    inputs = r.json()

    headers = ['pregnancies','glucose','bloodpressure','skinthickness','insulin','bmi','diabetespedigree','age','outcome']
    dictUIn = {}

    for key in headers:
        dictUIn[key] = {}

    for i in range(len(inputs)):
        for key in headers:
            dictUIn[key][i] = inputs[i][key]

    dataFrame = pd.read_json(json.dumps(dictUIn))

    # Características de entrada (Información de los campos del formulario)
    X = dataFrame.drop('outcome',axis=1)

    # Cracterísticas de salida ()
    y = dataFrame['outcome']

    # Separar la base de datos en 2 conjuntos (Entrenamiento y Prueba)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35)

    # Modelo
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    dt = DecisionTreeClassifier()

    # Entrenar el modelo
    dt.fit(X_train, y_train)

    # Exportar el modelo para usarlo en un servidor web con flask
    dump(dt,'../modelo.joblib')  # 64 bits

    #Regresar la salida del modelo
    return jsonify({"result": str(dt.score(X_test, y_test))})

# Predicción del modelo a partir de un formulario
@servidorWeb.route('/prediccion', methods=['POST'])
def prediccion():
    #Procesar datos de entrada 
    contenido = request.form
    
    datosEntrada = np.array([
            contenido['Pregnancies'],
            contenido['Glucose'],
            contenido['BloodPressure'],
            contenido['SkinThickness'],
            contenido['Insulin'],
            contenido['BMI'],
            contenido['DiabetesPedigreeFunction'],
            contenido['Age']
        ])
    #Utilizar el modelo
    resultado=dt.predict(datosEntrada.reshape(1,-1))
    #Regresar la salida del modelo
    return jsonify({"Resultado":str(resultado[0])})

#Envio de datos a través de JSON
@servidorWeb.route('/rePoblar', methods=['POST'])
def rePoblar():

    file = open('../diabetes.csv', 'r')

    line = file.readline()[0:-1].lower()
    # headers = line.split(',')
    headers = ['pregnancies','glucose','bloodpressure','skinthickness','insulin','bmi','diabetespedigree','age','outcome']
    line = file.readline()[0:-1]

    url = 'http://localhost:8080/Base/crearRegistro'
    while line != '':
        values = line.split(',')
        data = { "id" : 0}
        for header in headers:
            data[header] = values[headers.index(header)]
        # print(data)
        r = requests.post(url = url, data= data)
        line = file.readline()[0:-1]
    #Regresar la salida del modelo
    return jsonify({"Base":"repoblada"})

if __name__ == '__main__':
    servidorWeb.run(debug=False,host='0.0.0.0',port='8081')