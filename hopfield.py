import numpy as npy
import re
import os
from PIL import Image

def MatrizaVector(x):
    m = x.shape[0]*x.shape[1]
    aux = npy.zeros(m)
    c = 0
    for i in range (x.shape[0]):
        for j in range (x.shape[1]):
            aux[c] = x[i,j]
            c += 1
    return aux

def matrizPeso(x):
    if len(x.shape) != 1:
        print("La entrada no es un vector")
        return
    else:
        w = npy.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range (len(x)):
                if i != j:
                    w[i,j] = x[i]*x[j]
                    w[j,i] = w[i,j]
    return w
        
def ImagenaMatriz(file, size, threshold=145):
    imagen = Image.open(file).convert(mode = "L") #Abrir la imagen y pasarla a blanco y negro
    imagen = imagen.resize(size)
    vectorimagen = npy.asarray(imagen,dtype=npy.uint8)
    x = npy.zeros(vectorimagen.shape,dtype=npy.float64)
    x[vectorimagen > threshold] = 1
    x[vectorimagen <= threshold] = -1
    return x

def hopfield(entradas, size=(2,2), threshold = 60):
    print ("Creando matriz de peso")
    archivos = 0
    for path in entradas:
        x = ImagenaMatriz(file = path, size = size, threshold=threshold)
        x_vec = MatrizaVector(x)
        if archivos == 0:
            w = matrizPeso(x_vec)
            archivos = 1
        else:
            aux_w = matrizPeso(x_vec)
            w = w + aux_w
            archivos += 1
    print ("La matriz de pesos ha sido creada")
    print (npy.matrix(w))

directorio = os.getcwd()
direntrenamiento = []
path = directorio+"/entrenamiento/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-]*.jp[e]*g',i):
        direntrenamiento.append(path+i)

hopfield(entradas=direntrenamiento, size=(2,2), threshold= 60)