import pandas as pd
import tensorflow as tf
import custom_metrics as cm
import sklearn as sk
from tensorflow import keras
from tensorflow.python.keras import layers


def build_model(train_dataset):

    # Genera un modelo de red neuronal con las características especificadas en la función.

    # Parámetros:
    #       train_dataset (DataFrame): El dataframe con el conjunto de entrenamiento.

    # Devuelve:
    #       model (Sequential): La red neuronal compilada.

    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adagrad(lr=0.06)
    model.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'], )
    return model


def train_model(model, train_data, train_labels, epochs):

    # Entrena el modelo de red que le hayamos pasado por parámetro.

    # Parámetros:
    #       model (Sequential): El modelo compilado de la red neuronal que vayamos a entrenar.
    #       train_data (DataFrame): Dataframe con las variables predictoras para el modelo.
    #       train_labels (Series): Objeto Series de pandas con la variable objetivo.
    #       epochs (int): Valor entero con el número de épocas para entrenar la red.

    # Devuelve:
    #       hist (DataFrame): DataFrame con las métricas y épocas del entrenamiento.

    EPOCHS = epochs
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        train_data, train_labels,
        epochs=EPOCHS, validation_split=0.33, verbose=2, shuffle=True, callbacks=[early_stop])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    return hist


def KFold_train_model(model, train_data, train_labels, test_data, test_labels, epochs, folds, lin_lin):

    # Entrena el modelo de red pasado por parámetro por medio de validación cruzada.

    # Parámetros:
    #       model (Sequential): El modelo compilado de red neuronal que vayamos a entrenar.
    #       train_data (DataFrame): DataFrame con las variables predictoras del conjunto de entreanmiento del modelo.
    #       train_labels (Series): Objeto Series de pandas con la variable objetivo de los entrenamientos.
    #       test_data (DataFrame): DataFrame con las variables predictoras del conjunto de validación.
    #       test_labels (Series): Objeto Series de pandas con la variable objetivo de la validación.
    #       epochs (int): Valor entero con el número de épocas para entrenar la red.
    #       folds (int): Valor entero con el número de pliegues para hacer la validación cruzada.
    #       lin_lin (boolean): En caso de ser True o False evalúa los resultados teniendo en cuenta si el modelo es
    #                          lin-lin o log-lin respectivamente.

    # Devuelve:
    #       Nada.

    kf = sk.model_selection.KFold(n_splits=folds)
    EPOCHS = epochs

    for train, test in kf.split(train_data, train_labels):

        train_model(model, train_data.iloc[train], train_labels.iloc[train], EPOCHS)

        if lin_lin:
            cm.evaluate_lin_lin(train_data.iloc[test], train_labels.iloc[test], train_data.iloc[train],
                                train_labels.iloc[train], model)
        else:
            cm.evaluate_log_lin(train_data.iloc[test], train_labels.iloc[test], train_data.iloc[train],
                                train_labels.iloc[train], model)
    if lin_lin:
        cm.evaluate_lin_lin(test_data, test_labels, train_data, train_labels, model)
    else:
        cm.evaluate_log_lin(test_data, test_labels, train_data, train_labels, model)
