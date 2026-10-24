import numpy as np
import matplotlib
matplotlib.use('TkAgg') # Motor gráfico para WSL
import matplotlib.pyplot as plt

class PerceptronVisual:
    def __init__(self, learning_rate=0.1, epochs=25):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(2)
        self.bias = 0.0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def train_and_plot(self, X, y, title):
        # ¡Aquí están los pesos con los que estaban jugando!
        # Son un excelente punto de partida para ver cómo la IA los corrige.
        self.weights = np.array([0.7, -0.9]) 
        self.bias = -0.2

        plt.figure(figsize=(6, 6))

        for epoch in range(self.epochs):
            error_flag = False
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    error_flag = True

            # --- DIBUJAR LA GRÁFICA ---
            plt.clf()
            plt.title(f"{title} - Época {epoch + 1}") 
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            
            # Dibujar los puntos (rojo para 0, azul para 1)
            for i, xi in enumerate(X):
                color = 'blue' if y[i] == 1 else 'red'
                plt.scatter(xi[0], xi[1], color=color, s=100, zorder=5)

            # Dibujar la línea verde (frontera de decisión)
            if self.weights[1] != 0:
                x1_vals = np.array([-0.5, 1.5])
                x2_vals = -(self.weights[0] * x1_vals + self.bias) / self.weights[1]
                plt.plot(x1_vals, x2_vals, color='green', linestyle='--')

            plt.draw()
            plt.pause(0.8)

            if not error_flag:
                print(f"\n¡Éxito! Entrenamiento completado para {title} en la época {epoch + 1}.")
                break
        else:
            print(f"\n¡Límite alcanzado! El Perceptrón NO pudo resolver la {title} en {self.epochs} épocas.")

        plt.show()

if __name__ == "__main__":
    perceptron = PerceptronVisual(learning_rate=0.1, epochs=25)

    # --- COMPUERTA AND (SÍ ES LINEALMENTE SEPARABLE) ---
    X_datos = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_datos = np.array([0, 0, 0, 1]) 
    
    print("Ejecutando visualización para COMPUERTA AND...")
    perceptron.train_and_plot(X_datos, y_datos, "Compuerta AND")