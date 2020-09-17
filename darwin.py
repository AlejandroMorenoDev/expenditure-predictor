import tensorflow as tf
import keras.backend as kb
import random as rd
import expenditure_predictor as ep
from tensorflow import keras
from tensorflow.python.keras import layers
from sklearn.metrics import r2_score

best_score = 0
best_model = []


def fitness(model, test_dataset, test_labels):

    # Evalúa el modelo de red pasado por parámetro y además guarda una copia del mejor modelo y la mejor precisión hasta
    # el momento

    # Parámetros:
    #       model (Sequential): El modelo de red a evaluar.
    #       test_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de validación.
    #       train_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de entrenamiento.

    # Devuelve:
    #       r2(float): R2 resultante de evaluar el modelo.

    global best_score
    global best_model
    predicted_data = model.predict(test_dataset).flatten()
    r2 = r2_score(test_labels, predicted_data)
    print("R2 sobre validación: ", r2)
    if r2 >= best_score:
        best_score = r2
        best_model = model
    return r2


def create_population(train_dataset):

    # Crea la población de redes con distintas configuraciones para el algoritmo evolutivo.

    # Parámetros:
    #       train_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de entrenamiento.


    # Devuelve:
    #       population(list): Lista con los modelos para el algoritmo genético.

    optimizers = [tf.keras.optimizers.Adagrad(lr=rd.randrange(1, 300) / 1000),
                  tf.keras.optimizers.Adadelta(lr=rd.randrange(1, 300) / 1000),
                  tf.keras.optimizers.Adam(lr=rd.randrange(1, 300) / 1000),
                  tf.keras.optimizers.Adamax(lr=rd.randrange(1, 300) / 1000),
                  tf.keras.optimizers.RMSprop(lr=rd.randrange(1, 300) / 1000),
                  tf.keras.optimizers.Nadam(lr=rd.randrange(1, 300) / 1000)]

    population = []

    for optimizer in optimizers:
        inputs = tf.keras.Input(shape=(len(train_dataset.keys()),))

        # Nos encargamos de que el modelo vaya a tener mínimo una capa densa

        dense = layers.Dense(rd.randrange(20, 60), activation="relu")

        x = dense(inputs)

        # El modelo tendrá entre 2 y 10 capas densas elegidas al azar y entre 20 y 60 neuronas por capa.

        for n in range(rd.randrange(1, 10)):
            x = layers.Dense(rd.randrange(20, 60), activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name="model")
        model.compile(loss='mae',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'], )
        population.append(model)
    return population


def evolve(survivors, train_dataset):

    # Regenera la población a investigar y añade alguna modificación.

    # Parámetros:
    #       train_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de entrenamiento.


    # Devuelve:
    #       survivors(list): Lista con los 6 modelos que formarán la próxima generación.

    if len(survivors) == 0:
        return create_population(train_dataset)

    while len(survivors) < 6:

        inputs = tf.keras.Input(shape=(len(train_dataset.keys()),))

        # La primera capa de los "hijos" siempre será igual a la del mejor modelo.

        dense = best_model.layers[1]
        x = dense(inputs)

        n_layers = rd.randrange(1, len(best_model.layers) - 1)
        count = 0

        # El resto de capas del hijo será un subconjunto de las n primeras capas del mejor modelo.

        for layer in best_model.layers[2:-1]:
            if count == n_layers:
                break
            x = layers.Dense(layer.units)(x)
            count = count + 1

        outputs = layers.Dense(1)(x)
        child = keras.Model(inputs=inputs, outputs=outputs, name="model")
        child.compile(loss='mae',
                      optimizer=best_model.optimizer,
                      metrics=['mae', 'mse'], )
        kb.set_value(child.optimizer.lr, kb.eval(survivors[0].optimizer.lr))
        # Los hijos siempre mutan su learning rate
        child = mutate(child)
        survivors.append(child)
    return survivors


def mutate(model):

    # Muta el learning rate del modelo pasado por parámetro.

    # Parámetros:
    #       model (Sequential): Modelo de red que se va a mutar.

    # Devuelve:
    #       model (Sequential): Modelo de red con la mutación aplicada.

    kb.set_value(model.optimizer.lr, rd.randrange(1, 300) / 1000)
    return model


def natural_selection(subjects, train_dataset, train_labels, test_dataset, test_labels, generations):

    # Filtra los mejores modelos de cada generación

    # Parámetros:
    #       subjects (list): Lista con los modelos de red neuronal que van a formar la población de estudio.
    #       train_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de entrenamiento.
    #       train_labels (Series): Objeto Series de pandas con la variable objetivo de los entrenamientos.
    #       test_dataset (DataFrame): DataFrame con las variables predictoras del conjunto de validación.
    #       test_labels (Series): Objeto Series de pandas con la variable objetivo de la validación.
    #       generations (int): Valor entero con el número de generaciones a ejecutar.
    
    # Devuelve:
    #       best_model (Sequential): Mejor modelo diseñado hasta el momento.
    #       best_score (float): Valor con el coeficiente de determinación del mejor modelo

    global best_score
    global best_model
    count = 1
    for g in range(generations):
        print("##------------------------##")
        print("Entrenando generación: ", count)
        print("##------------------------##")
        for subject in subjects:
            ep.KFold_train_model(subject, train_dataset, train_labels, test_dataset, test_labels, 100, 5, True)
        print("Mejor R2 hasta el momento: ", best_score)

        # Filtramos tomando como referencia el mejor modelo que hayamos tenido hasta la fecha, permitiendo un
        # margen del 0,01 en la precisión.

        survivors = [subject for subject in subjects if fitness(subject, test_dataset, test_labels) >= best_score - 0.01]

        survivors = sorted(survivors, key=lambda survivor: fitness(survivor, test_dataset, test_labels), reverse=True)
        subjects = evolve(survivors, train_dataset)
        count = count+1
    return best_model, best_score


