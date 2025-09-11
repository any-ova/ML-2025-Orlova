import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class LinearRegressorGD:
    """
    Линейная регрессия с использованием Gradient Descent
    """

    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
        Конструктор класса

        Параметры:
            learning_rate (float): Скорость обучения
            n_iter (int): Количество итераций градиентного спуска
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.coeff = None
        self.intercept = None

    def compute_gradients(self, X, y_true, y_pred):
        "Подсчёт градиента"
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        n_samples = len(y_true)
        error = y_pred - y_true
        grad_coeff = (2 / n_samples) * (X.T @ error)
        grad_intercept = (2 / n_samples) * np.sum(error)
        return grad_coeff, grad_intercept

    def fit(self, X, y):
        """
                Обучение модели на обучающей выборке с использованием
                градиентного спуска

                Параметры:
                    X (np.ndarray): Матрица признаков размера (n_samples, n_features)
                    y (np.ndarray): Вектор таргета длины n_samples
                """
        n_samples, n_features = X.shape
        np.random.seed(42)
        self.coeff = np.random.randn(n_features) * 0.01
        self.intercept = 0.0

        for i in range(self.n_iter):
            y_pred = self.predict(X)
            dc, di = self.compute_gradients(X, y, y_pred)
            self.coeff -= self.learning_rate * dc
            self.intercept -= self.learning_rate * di

        return self

    def predict(self, X):
        """
        Получение предсказаний обученной модели

        Параметры:
            X (np.ndarray): Матрица признаков

        Возвращает:
            np.ndarray: Предсказание для каждого элемента из X
        """
        if self.coeff is None:
            raise ValueError("Модель не обучена")
        y_pred = X @ self.coeff + self.intercept
        return y_pred

    def get_params(self):
        """
        Возвращает обученные параметры модели
        """
        return (self.coeff, self.intercept)


class MLPRegressor:
    """
    Многослойный перцептрон (MLP) для задачи регрессии, использующий алгоритм
    обратного распространения ошибки
    """

    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, n_iter=100):
        """
        Конструктор класса

        Параметры:
            hidden_layer_sizes (tuple): Кортеж, определяющий архитектуру
        скрытых слоев. Например (100, 10) - два скрытых слоя, размером 100 и 10
        нейронов, соответственно
            learning_rate (float): Скорость обучения
            n_iter (int): Количество итераций градиентного спуска
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.n_iter=n_iter
        self.coeff=[]
        self.biases =[]
        self.layers=[]

    def initialize_weights(self, n_features, n_output=1):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_output]

        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]) #нормальное распределение с масштабированием
            b = np.zeros((1, layer_sizes[i+1]))
            self.coeff.append(W)
            self.biases.append(b)
    def sigmoid(self,z):
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))
    def sigmoid_proizvod(self,z):
        return z*(1-z)
    def forward(self, X):
        """
        Реализация forward pass

        Параметры:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)

        Возвращает:
            np.ndarray: Предсказания модели
        """
        self.layers = [X]

        for i in range(len(self.coeff) - 1):
            z = np.dot(self.layers[-1], self.coeff[i]) + self.biases[i]
            self.layers.append(self.sigmoid(z))

        output = np.dot(self.layers[-1], self.coeff[-1]) + self.biases[-1]
        self.layers.append(output)
        return output

    def backward(self, X, y):
        """
        Реализация backward pass

        Возвращает:
            X (np.ndarray): Матрица признаков размера (n_samples, n_features)
            y (np.ndarray): Вектор таргета длины n_samples
        """
        n_samples = X.shape[0]
        dZ = 2 * (self.layers[-1] - y)                # градиенты для последнего слоя
        dc = [None] * len(self.coeff)
        db = [None] * len(self.biases)

        for i in reversed(range(len(self.coeff))):
            dc[i] = np.dot(self.layers[i].T, dZ)/ n_samples
            db[i] = np.sum(dZ, axis=0, keepdims=True)/ n_samples

            if i > 0:
                dA_prev = np.dot(dZ, self.coeff[i].T)                  # градиент для предыдущего слоя
                dZ = dA_prev * self.sigmoid_proizvod(self.layers[i])

        for i in range(len(self.coeff)):
            self.coeff[i] -= self.learning_rate * dc[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X, y):
        self.initialize_weights(X.shape[1], 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        y_normalized = (y - self.y_mean) / (self.y_std + 1e-8)

        for i in range(self.n_iter):
            y_pred = self.forward(X)
            self.backward(X, y_normalized)

            # if i % 100 == 0 or i == self.n_iter - 1:
            #     mse = np.mean((y_pred - y_normalized) ** 2)
            #     print(f"Iteration {i}, MSE: {mse:.6f}", flush=True)

        return self

    def predict(self, X):
        """
        Получение предсказаний обученной модели

        Параметры:
            X (np.ndarray): Матрица признаков

        Возвращает:
            np.ndarray: Предсказание для каждого элемента из X
        """
        y_pred_normalized = self.forward(X)
        y_pred = y_pred_normalized * self.y_std + self.y_mean

        return y_pred.flatten()









diamonds_df = pd.read_csv('diamonds.csv')

features = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
target = ['price']

cut_transform = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
clarity_transform = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
color_transform = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}

diamonds_df['cut'] = diamonds_df['cut'].apply(lambda x: cut_transform.get(x))
diamonds_df['color'] = diamonds_df['color'].apply(lambda x: color_transform.get(x))
diamonds_df['clarity'] = diamonds_df['clarity'].apply(lambda x: clarity_transform.get(x))

X = diamonds_df[features].copy().values
y = diamonds_df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47, test_size=0.3)

scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

reg2 = LinearRegressorGD(learning_rate=0.01, n_iter=5000)
reg2.fit(X_train, y_train_scaled)

y_pred_scaled = reg2.predict(X_test)
y_pred2 = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

score2 = r2_score(y_test, y_pred2)
mae = mean_absolute_error(y_test.flatten(), y_pred2)
mse = mean_squared_error(y_test.flatten(), y_pred2)
rmse = np.sqrt(mse)


print("1 zadanie")
print(f"R2 Score: {score2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


print("zadanie 2")

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate=0.01, n_iter=5000)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)

score_mlp = r2_score(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)

print(f"R2 Score: {score_mlp:.4f}")
print(f"MAE: {mae_mlp:.2f}")
print(f"RMSE: {rmse_mlp:.2f}")