import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self) -> None:
        self.linearReg = LinearRegression()
        self.standardS = StandardScaler()
        self.randForestReg = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def load_data(self, filePath):
        df = pd.read_csv(filePath)
        # Проверка данных
        print(df.head())
        print(df.info())
        print(df.describe())
        print(df.isnull().sum())

        self.X = df.drop('output', axis=1)  # Признаки
        self.y = df['output']  # Целевая переменная
    
    def slit_data(self):
        self.X_train, X_test, self.y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Масштабирование данных
        self.X_train = self.standardS.fit_transform(self.X_train)
        X_test = self.standardS.transform(X_test)
        return X_test, y_test
    
    def train(self):
        self.linearReg.fit(self.X_train, self.y_train)
        self.randForestReg.fit(self.X_train, self.y_train)

    def linearRegress(self,X,Y):
        # Предсказание
        y_pred_lin = self.linearReg.predict(X)


        # Вычисление ROC-кривой
        fpr, tpr, thresholds = roc_curve(Y, y_pred_lin)
        roc_auc = auc(fpr, tpr)
        print(f'Точноть модели "LinearRegression":{roc_auc:.2f}')
        # Построение ROC-кривой
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC-кривая (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Линия случайного предсказания
        plt.xlabel('Ложноположительная ставка')
        plt.ylabel('Истинноположительная ставка')
        plt.title('ROC-кривая')
        plt.legend()
        plt.grid()
        plt.show()

        plt.scatter(Y, y_pred_lin, alpha=0.5)
        plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--')  # линия y=x
        plt.xlabel('Истинные значения')
        plt.ylabel('Предсказанные значения')
        plt.title('Сравнение предсказанных и истинных значений')
        plt.show()

        residuals = Y - y_pred_lin
        plt.hist(residuals, bins=30)
        plt.title('Распределение остатков')
        plt.xlabel('Остатки')
        plt.ylabel('Частота')
        plt.show()

    def forestRand(self, X, Y):
        # Предсказание
        y_pred_rf = self.randForestReg.predict(X)

        # Оценка модели
        print("Случайный Лес:")
        print("MSE:", mean_squared_error(Y, y_pred_rf))
        print("R^2:", r2_score(Y, y_pred_rf))
    
    def modelLasso(self, X, Y, A):
        self.lasso = Lasso(alpha=A)
        self.lasso.fit(self.X_train, self.y_train)

        # Предсказание
        y_pred = self.lasso.predict(X)

        # Оценка производительности
        mse = mean_squared_error(Y, y_pred)
        r2 = r2_score(Y, y_pred)

        print(f'Среднеквадратическая ошибка в Lasso: {mse}')
        print(f'R^2 Счёт в Lasso: {r2}')

    def alpha(self):
        # Определение параметров для поиска
        param_grid = {'alpha': np.logspace(-4, 0, 50)}  # диапазон alpha от 0.0001 до 1

        # Создание модели Lasso
        lasso = Lasso()

        # Создание GridSearchCV
        grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)

        # Лучшее значение alpha
        best_alpha = grid_search.best_params_['alpha']
        print(f'Лучшее значение alpha: {best_alpha}')
        return best_alpha
