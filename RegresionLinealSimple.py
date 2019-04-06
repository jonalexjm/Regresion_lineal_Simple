#importamos las librerias
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


##################Preparamos la data#################
#se importan las librerias a utilizar
boston = datasets.load_boston()
print(boston)
print()

##################Entendimiento de  la data#################
#Verifico la informacion contenida en el dataset
print('informacion  del dataset: ')
print(boston.keys())
print()

#Verifico las caracteristicas en el dataset
print('Caracteristicas del dataset: ')
print(boston.DESCR)
print()

#verifico la cantidad de datos que hay en los dataset
print('Cantidad de datos: ')
print(boston.data.shape)
print()

#verifico nombre de columnas
print('Nombres columnas: ')
print(boston.feature_names)
print()

##################PREPARAR LA DATA REGRESION LINEAL SIMPLE#################

#selecionamos solamente la columna 5 del data set
X = boston.data[:, np.newaxis, 5]

#Defino los datos correspondientes a las etiquetas
y = boston.target



#Graficamos los datos correspondientes
plt.scatter(X, y)
plt.xlabel('Numero de habitaciones')
plt.ylabel('Valor medio')
#plt.show()

########### IMPLEMENTACION DE REGRESION LINEAL SIMPLE #############

from sklearn.model_selection import train_test_split

#Separo los datos del "train " en entrenamiento y prueba para probar los algoritmos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Definimos el algoritmo a utilizar

lr = linear_model.LinearRegression()

#Realizo una prediccion

lr.fit(X_train, y_train)

#realizo una prediccion

Y_pred = lr.predict(X_test)

#Graficamos los datos junto con el modelo

plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresion Lineal Simple')
plt.xlabel('numero de habitaciones')
plt.ylabel('valor medio')
plt.show()

print()
print('Datos del modelo regresion lineal simple')
print('')
print('valor de la pendiente o coeficient "a"')
print(lr.coef_)
print('varlos de interseccion o coeficiente "b"')
print(lr.intercept_)

print('La ecuacion del modelo es igual a :')
print('y =', lr.coef_, 'x ', lr.intercept_)

print()
print('Precision del modelo es igual a:')
print(lr.score(X_train, y_train))

