### python -m venv venv
### source venv/bin/activate  # для Windows: venv\Scripts\activate

# Устанавливаем зависимости 
### pip install scikit-learn pandas matplotlib seaborn 

# В этой модели машиного обучения мы будем использовать dataset - Набор данных для анализа и прогнозирования сердечного приступа

# Вывод:
Используя модель LinearRegression в обучении и тестировании мы достигли высоких результатов в предсказании в 0.94.


Использование "Случайнй Лес" - не рекомендально, показывает низские результаты в значении - 0.53. Возможно данные не подходят для этого метода предсказания или их не достаточно.

Использование модели Lasso по ленейной регресси набрало 0.54, что указывает на  умеренную предсказательную способность  модели, что является давольно сомнительным результатом даже с подбором оптимального значения alpha

Точноть модели "LinearRegression":0.94
Лучшее значение alpha: 0.010985411419875584
Среднеквадратическая ошибка в Lasso: 0.11497070514044272
R^2 Счёт в Lasso: 0.5390021618237206
