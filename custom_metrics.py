import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


def adjusted_r2(r2, test_dataset, train_dataset):

    # Calcula el R2 ajustado en función de los parámetros que recibe.

    # Parámetros:
    #       r2 (float): Valor del R2 del modelo para el que se calcula la métrica.
    #       test_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de validación.
    #       train_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de entrenamiento.

    # Devuelve:
    #      (float): R2 ajustado calculado sobre los parámetros.

    return 1 - (test_dataset.shape[0] - 1) / (test_dataset.shape[0] - train_dataset.shape[1] - 1) * (1 - r2)


def MAPE(predict, target):

    # Calcula el MAPE en función de los parámetros que recibe.

    # Parámetros:
    #       predict (numpy.array): Contiene los valores de predicción devueltos por el modelo
    #       target (numpy.array o Series): Contiene los valores reales para comparar con las predicciones.

    # Devuelve:
    #       (float): MAPE calculado sobre los parámetros.
    return (abs((target - predict) / target).mean()) * 100


def plot_history(hist):

    # Muestra una gráfica con la evolución de los errores de entrenamiento y validación.

    # Parámetros:
    #     hist(DataFrame): DataFrame con las métricas de entrenamiento y validación por épocas.

    # Devuelve:
    #     Nada.

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [GASTO]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
            label = 'Val Error')
    plt.ylim([400, 800])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$GASTO^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
            label = 'Val Error')
    plt.ylim([0, 100000])
    plt.legend()
    plt.show()


def plot_regression(predictions, test_labels):

    # Muestra una gráfica comparando los valores predecidos por el modelo frente a los valores reales.

    # Parámetros:
    #       predict (numpy.array): Contiene los valores de predicción devueltos por el modelo
    #       target (numpy.array o Series): Contiene los valores reales para comparar con las predicciones.

    # Devuelve:
    #       Nada.

    plt.scatter(test_labels, predictions)
    plt.xlabel('Valores Originales')
    plt.ylabel('Valores Predichos')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    z = np.polyfit(test_labels, predictions, 1)
    p = np.poly1d(z)
    plt.plot(test_labels, p(test_labels), "r--")
    plt.show()


def evaluate_lin_lin(test_dataset, test_labels, train_dataset, train_labels, model):

    # Devuelve el r2 calculado sobre el conjunto de validación, además imprime todas las métricas del modelo lin-lin
    # a evaluar.

    # Parámetros:
    #       test_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de validación.
    #       test_labels (Series):
    #       train_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de entrenamiento.
    #       train_labels (Series): Objeto Series de pandas con la variable objetivo de los entrenamientos.
    #       test_labels (Series): Objeto Series de pandas con la variable objetivo de la validación.
    #       model: Cualquier modelo predictivo de sklearn o keras que se quiera evaluar.

    # Devuelve:
    #       r2(float): R2 resultante de evaluar el modelo lin-lin

    predicted_data = model.predict(train_dataset).flatten()
    r2 = r2_score(train_labels, predicted_data)
    print("----Resultados sobre TRAIN----")
    print("R2: ", r2)
    print("Adjusted R2: ", adjusted_r2(r2, train_dataset, train_dataset))
    print('MAPE: ' + str(MAPE(predicted_data, train_labels)))

    predicted_data = model.predict(test_dataset).flatten()
    r2 = r2_score(test_labels, predicted_data)
    print("----Resultados sobre TEST----")
    print("R2: ", r2)
    print("Adjusted R2: ", adjusted_r2(r2, test_dataset, train_dataset))
    print('MAPE: ' + str(MAPE(predicted_data, test_labels)))
    plot_regression(predicted_data, test_labels)
    return r2


def evaluate_log_lin(test_dataset, test_labels, train_dataset, train_labels, model):

    # Devuelve el r2 calculado sobre el conjunto de validación, además imprime todas las métricas del modelo log-lin
    # a evaluar.

    # Parámetros:
    #       test_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de validación.
    #       test_labels (Series):
    #       train_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de entrenamiento.
    #       train_labels (Series): Objeto Series de pandas con la variable objetivo de los entrenamientos.
    #       test_labels (Series): Objeto Series de pandas con la variable objetivo de la validación.
    #       model: Cualquier modelo predictivo de sklearn o keras que se quiera evaluar.

    # Devuelve:
    #       r2(float): R2 resultante de evaluar el modelo.

    predicted_data = model.predict(train_dataset).flatten()
    r2 = r2_score(np.exp(train_labels), np.exp(predicted_data))
    print("----Resultados sobre TRAIN----")
    print("R2: ", r2)
    print("Adjusted R2: ", adjusted_r2(r2, train_dataset, train_dataset))
    print('MAPE: ' + str(MAPE(np.exp(predicted_data), np.exp(train_labels))))

    predicted_data = model.predict(test_dataset).flatten()
    r2 = r2_score(np.exp(test_labels), np.exp(predicted_data))
    print("----Resultados sobre TEST----")
    print("R2: ", r2)
    print("Adjusted R2: ", adjusted_r2(r2, test_dataset, train_dataset))
    print('MAPE: ' + str(MAPE(np.exp(predicted_data), np.exp(test_labels))))
    plot_regression(predicted_data, test_labels)
    return r2