import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import kstest_normal


def normality_test(y):

    # Realiza un test de normalidad de Kolmogorov-Smirnnof sobre la variable endógena y muestra una gráfica de
    # distribución

    # Parámetros:
    #       y (Series): Objeto Series de pandas con la variable endógena.

    # Devuelve:
    #       Nada.

    normality_test = kstest_normal(y, dist='norm')
    stats = ['KS Stat', 'KS-Test p-value']

    print("---Test de Kolmogorov-Smirnnof---")
    print(dict(zip(stats, normality_test)))
    if(normality_test[1] > 0.05):
        print("Aceptamos la hipótesis nula, ergo la variable se distribuye según una normal.")
    else:
        print("Fallamos al aceptar la hipótesis nula, ergo la variable no se distribuye según una normal.")
    sns.distplot(y)
    plt.show()


def heteroskedasticity_test(X, form):

    # Realiza un test de White y un test de Breusch-Pagan para determinar la heterocedasticidad del modelo.

    # Parámetros:
    #       X (DataFrame): Dataframe con las variables exógenas.
    #       form (str): String con la fórmula a  utilizar para la regresión.

    # Devuelve:
    #       Nada.

    expenditure_model = ols(formula=form, data=X).fit()
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    white_test = het_white(expenditure_model.resid, expenditure_model.model.exog)
    bp_test = het_breuschpagan(expenditure_model.resid, expenditure_model.model.exog)
    print("---Test de White---")
    print(dict(zip(labels, white_test)))

    print("---Test de BP---")
    print(dict(zip(labels, bp_test)))



