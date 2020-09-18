import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE


def corr_selection(X, threshold, y):

    # Realiza la selección de características filtrando por correlación de los predictores.

    # Parámetros:
    #       X (DataFrame): DataFrame con todas las variables predictoras.
    #       threshold (float): Valor real con el umbral a tener en cuenta para filtrar.
    #       y (Series): Objeto Series de pandas con la variable endógena.

    # Devuelve:
    #       Nada.

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        start = datetime.datetime.now()
        print(abs(X.corr()[y]).sort_values(ascending=False))
        end = datetime.datetime.now()
        print("La selección ha tardado: ", end - start)
        if threshold is not None:
            print("###############################################")
            print("Variables con correlación mayor o igual al umbral: ", threshold)
            print(X.loc[:, abs(X.corr()[y]) >= threshold].columns)


def rfe_selection(X, y, model):

    # Realiza la selección de características por medio de un eliminado recursivo de características..

    # Parámetros:
    #       X (DataFrame): DataFrame con todas las variables predictoras.
    #       y (Series): Objeto Series de pandas con la variable endógena.
    #       model: Cualquier modelo de sklearn que se vaya a utilizar como referencia para la selección.

    # Devuelve:
    #       Nada.

    rfe = RFE(model, 12, step=1, verbose=1)
    rfe = rfe.fit(X, y)
    rfe.score(X, y)
    X = X * rfe.support_
    X = X.loc[:, (X != 0).any(axis=0)]
    X = pd.concat([X, y], axis=1, sort=False)
    print(X.columns)
    show_correlation(X)


def rf_selection(X, y, jobs):

    # Realiza la selección de características utilizando un random forest.

    # Parámetros:
    #       X (DataFrame): DataFrame con todas las variables predictoras.
    #       y (Series): Objeto Series de pandas con la variable endógena.
    #       jobs(int): Número .

    # Devuelve:
    #       Nada.

    regressor = RandomForestRegressor(100, criterion="mae", verbose=2, n_jobs=jobs)
    regressor.fit(X, y)
    for feature in sorted(zip(X.columns, regressor.feature_importances_), key=lambda x: x[1]):
        print(feature)
    return regressor


def check_vif(X):

    # Muestra el factor de inflación de la varianza de los predictores pasados por parámetro.

    # Parámetros:
    #       X (DataFrame): DataFrame con todas las variables predictoras.

    # Devuelve:
    #       Nada.

    vif = pd.DataFrame()
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['variables'] = X.columns
    print(vif)


def show_correlation(X):

    # Muestra un mapa de calor con la matriz de correlaciones de los predictores pasados por parámetro.

    # Parámetros:
    #       X (DataFrame): DataFrame con todas las variables predictoras.

    # Devuelve:
    #       Nada.

    correlation = X.corr(method="pearson")
    plt.figure(figsize=(27, 8))
    sns.heatmap(correlation, annot=True, linewidths=0, vmin=-0.75, vmax=0.75, cmap="RdBu_r")
    plt.show()


def show_pca(features):

    # Realiza el PCA de las variables del DataFrame pasado por parámetro.

    # Parámetros: features (DataFrame): DataFrame con las variables a utilizar en el PCA.

    # Devuelve:
    #       Nada.
    pca = PCA()
    pca.fit(features)
    pca_result = pca.transform(features)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=3)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Porcentaje de Varianza Explicada')
    plt.xlabel('Componente Principal')
    plt.title('Gráfica')
    plt.show()
    loading_scores = pd.Series(pca.components_[0], index=features.columns)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    print(sorted_loading_scores)
    top_10_variables = sorted_loading_scores[0:10].index
    print(top_10_variables)
    pca_df = pd.DataFrame(pca_result, index=features.index, columns=labels)
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('Gráfico PCA')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    plt.show()

