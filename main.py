from fn.model import Model

model = Model()

model.load_data('./data/heart.csv')

x_test, y_test = model.slit_data()

model.train()

model.linearRegress(x_test, y_test)

model.modelLasso(x_test, y_test, model.alpha())

# использование  метода случайный лес
# model.forestRand(x_test, y_test)

# MSE (Mean Squared Error): Это средняя квадратичная ошибка, которая показывает, насколько предсказанные значения отличаются от реальных. Чем меньше значение, тем лучше модель.
# R^2 (Коэффициент детерминации): Это мера того, насколько хорошо модель объясняет вариацию целевой переменной. Значение R^2 близкое к 1 указывает на хорошее соответствие модели.  