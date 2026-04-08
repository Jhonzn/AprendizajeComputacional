import numpy as np
import pandas as pd

# -----------------------------
# CARGAR DATASET DESDE CSV
# -----------------------------
data = pd.read_csv("datos_aviones.csv", sep=";")

# Entradas y salidas
X = data[["combustible", "clima", "trafico"]].values
y = data["salida"].values

# Normalizar combustible
X[:,0] = X[:,0] / 100


# -----------------------------
# PERCEPTRÓN
# -----------------------------
class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                linear = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if linear >= 0 else 0

                error = y[i] - y_pred
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

    def predict(self, x):
        linear = np.dot(x, self.weights) + self.bias
        return 1 if linear >= 0 else 0


# -----------------------------
# VALIDACIONES
# -----------------------------
def validar_combustible(valor):
    return 10 <= valor <= 100

def validar_binario(valor):
    return valor in [0, 1]


# -----------------------------
# ENTRENAR MODELO
# -----------------------------
modelo = Perceptron()
modelo.train(X, y)


# -----------------------------
# SISTEMA INTERACTIVO
# -----------------------------
def sistema():
    try:
        combustible = float(input("Ingrese combustible (%): "))
        if not validar_combustible(combustible):
            print("Error: Combustible debe estar entre 10 y 100")
            return

        clima = int(input("¿Clima favorable? (1/0): "))
        if not validar_binario(clima):
            print("Error: Clima solo puede ser 0 o 1")
            return

        trafico = int(input("¿Tráfico alto? (1/0): "))
        if not validar_binario(trafico):
            print("Error: Tráfico solo puede ser 0 o 1")
            return

        entrada = np.array([combustible / 100, clima, trafico])
        resultado = modelo.predict(entrada)

        print("\nResultado:")
        if resultado == 1:
            print("Avión autorizado para aterrizar/despegar")
        else:
            print("Avión debe esperar")

    except:
        print("Error: Entrada inválida")


# -----------------------------
# EJECUTAR
# -----------------------------
sistema()