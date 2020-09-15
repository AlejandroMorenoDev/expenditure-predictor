import pandas as pd
import numpy as np


def load_dataset(filename, separator, decimal, predictors, dummies, desglose):

    # Carga el fichero csv en un DataFrame y elimina observaciones y variables innecesarias.

    # Parámetros:
    #       filename (str): String con la ruta del fichero ".csv" a cargar.
    #       separator (char): Caracter separador de las filas del fichero.
    #       decimal (char): Caracter utilizado para marcar los decimales del fichero.
    #       predictors (list): Lista con las variables predictoras que queremos cargar. Si  queremos cargar
    #                          completamente el dataset deberemos especificarlo como valor None.
    #       dummies (boolean): Indica a la función si debe hacer la conversión de categóricas a dummies o no.
    #       desglose (boolean): Indica a la función si debe preparar u omitir las variables desglose del gasto.

    # Devuelve:
    #       X (DataFrame): DataFrame con nuestro dataset cargado.

    df = pd.read_csv(filename, sep=separator, decimal=decimal)

    # Eliminamos residentes y turistas de paso.

    df.drop(df[df['PROPOSITO'] == '2'].index, inplace=True)
    df.drop(df[df['PROPOSITO'] == '1'].index, inplace=True)
    df.drop(df[df['NOCHES'] > 31].index, inplace=True)

    if desglose is False:
        for col in df:
            if col.startswith('DESGLOSE_'):
                df = df.drop(col, axis=1)
    else:
        df = get_absolute(df)

    if dummies:
        df = pd.get_dummies(df)

    if predictors is not None:
        X = df.loc[:, predictors]
    else:
        X = df

    # Si el turista NO ha contratado un paquete el coste del vuelo y del alojamiento estarán en las variables
    # COSTE_ALOJ_EUROS Y COSTE_VUELOS_EUROS, en caso contrario estará en COSTE_PAQUETE_EUROS. El gasto total se
    # construirá como la suma de los gastos de esa partida y el gasto en consumo en destino (GASTO_EUROS).


    X['GASTO'] = np.where(df['COSTE_PAQUETE_EUROS'] > 0, df.COSTE_PAQUETE_EUROS + df.GASTO_EUROS,
                           df.COSTE_VUELOS_EUROS + df.COSTE_ALOJ_EUROS + df.GASTO_EUROS)

    # Eliminamos observaciones que no tengan gasto alguno.

    X.drop(X[X['GASTO'] == 0].index, inplace=True)

    for col in X.columns:
        # Eliminamos variables que no nos interesan para predecir el gasto.
        if col.startswith('INTERNET_') or col.startswith('CONEXION_') or col.endswith('NO_PROCEDE') or \
                col.startswith('IMPORTANCIA'):
            X.pop(col)
            continue
        # Eliminamos columnas y obsrevaciones con valores vacíos en las variables categóricas.
        if col.endswith('NO_SABE') or col.endswith('NO_CONTESTA'):
            X = X.loc[X[col] == 0]
            X.pop(col)

    # Eliminamos los outliers que existan sobre la variable GASTO
    X = remove_outliers(X)
    X.sample(frac=1)
    return X


''' 
remove_outliers(x)

Parámetros:
@x -> dataset a tratar

Utilidades:
1) Eliminar los outliers del dataset por recorrido intercuartílico

Devuele:
@x -> dataset sin outliers
'''


def remove_outliers(x):

    # Elimina los outliers de la variable endógena según recorrido intercuartílico.

    # Parámetros:
    #       x(DataFrame): Dataframe con el dataset cargado.

    # Devuelve:
    #       x(DataFrame): DataFrame sin outliers en la variable endógena.

    y = x['GASTO']
    q1 = y.quantile(.25)
    q3 = y.quantile(.75)
    IQR = (q3 - q1) * 1.5
    x = x[(y < q3 + IQR)]
    x = x[(y > q1 - IQR)]
    return x


def get_absolute(x):

    # Convierte a valor absoluto las partidas de DESGLOSE que están en valor relativo.

    # Parámetros:
    #       x(DataFrame): DataFrame con el dataset sobre el que hacer la conversión.

    # Devuelve:
    #       x(DataFrame): DataFrame con las variables DESGLOSE en valores absolutos.

    for col in x:
        if col.startswith('DESGLOSE_'):
            x[col] = x.GASTO_EUROS * x[col] / 100
    return x


def split_dataset(x):

    # Divide el DataFrame pasado por parámetro en los subconjuntos de validación y entrenamiento.

    # Parámetros:
    #       x (DataFrame): DataFrame con el dataset a dividir.

    # Devuelve:
    #       train (DataFrame): DataFrame con las observaciones para entrenamiento.
    #       test (DataFrame): DataFrame con las observaciones para validación.

    train = x.sample(frac=0.75, random_state=0)
    test = x.drop(train.index)
    return train, test

