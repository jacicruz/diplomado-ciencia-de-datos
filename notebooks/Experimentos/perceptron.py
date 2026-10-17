import os
import time
import numpy as np
import matplotlib
matplotlib.use(os.environ.get("MPL_BACKEND", "QtAgg"))  # pip install PyQt6
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class MLPVisual:
    """Perceptrón multicapa 2-2-1 (sigmoide) entrenado con backpropagation."""

    def __init__(self, learning_rate=1.0, epochs=6000, hidden=2, seed=1):
        self.lr = learning_rate
        self.epochs = epochs
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1, (2, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1, (hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.h = sigmoid(X @ self.W1 + self.b1)
        self.out = sigmoid(self.h @ self.W2 + self.b2)
        return self.out

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int).ravel()

    def step(self, X, y):
        y = y.reshape(-1, 1)
        out = self.forward(X)
        # Backpropagation (error cuadrático medio)
        d_out = (out - y) * out * (1 - out)
        d_h = (d_out @ self.W2.T) * self.h * (1 - self.h)
        self.W2 -= self.lr * self.h.T @ d_out
        self.b2 -= self.lr * d_out.sum(axis=0)
        self.W1 -= self.lr * X.T @ d_h
        self.b1 -= self.lr * d_h.sum(axis=0)
        return float(np.mean((out - y) ** 2))

    def draw(self, ax, X, y, title, epoch, loss):
        ax.clear()
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = self.forward(grid).reshape(xx.shape)

        # Región de clasificación (azul = clase 1, rojo = clase 0)
        ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=['#ffcccc', '#cce0ff'], alpha=0.7)
        ax.contour(xx, yy, zz, levels=[0.5], colors='k', linewidths=2)

        # Rectas de las neuronas ocultas
        x_vals = np.array([-0.5, 1.5])
        for j in range(self.W1.shape[1]):
            w1, w2 = self.W1[:, j]
            if abs(w2) > 1e-9:
                y_vals = (-w1 * x_vals - self.b1[j]) / w2
                ax.plot(x_vals, y_vals, '--', linewidth=1.5, label=f'Neurona oculta {j+1}')

        for i, s in enumerate(X):
            color = 'blue' if y[i] == 1 else 'red'
            marker = 'o' if y[i] == 1 else 'x'
            ax.scatter(s[0], s[1], color=color, s=150, marker=marker, linewidths=3, zorder=5)
            ax.text(s[0] + 0.05, s[1], f"({s[0]},{s[1]})={y[i]}", fontsize=10)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_title(f"{title}\nÉpoca {epoch} | Error={loss:.4f}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend(loc='upper right', fontsize=8)

    def train_and_plot(self, X, y, title, plot_every=100):
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

        for epoch in range(1, self.epochs + 1):
            loss = self.step(X, y)
            converged = np.array_equal(self.predict(X), y) and loss < 0.01

            if epoch == 1 or epoch % plot_every == 0 or converged:
                self.draw(ax, X, y, title, epoch, loss)
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.15)

            if converged:
                print(f"[{title}] ¡Convergencia alcanzada en la época {epoch}!")
                break

        print("Predicciones:")
        for xi, p in zip(X, self.predict(X)):
            print(f"  {xi} -> {p}")

        plt.ioff()
        time.sleep(1.5)
        plt.show()  # mantener la ventana abierta para ver el resultado final


# --- DATOS ENTRADA ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

if __name__ == "__main__":
    mlp = MLPVisual(learning_rate=1.0, epochs=6000, hidden=2, seed=1)
    print("\nEjecutando visualización para COMPUERTA XOR con PERCEPTRÓN MULTICAPA...")
    mlp.train_and_plot(X, y_xor, "Compuerta XOR - Perceptrón Multicapa (2-2-1)")
