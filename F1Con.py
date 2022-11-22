from os import listdir
from os.path import isfile,isdir, join
import numpy
import datetime
import sklearn
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop


ih, iw = 250, 250 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales

#Directorios de los datos
train_dir = 'Data/archive/img_align_celeba' #directorio de entrenamiento
test_dir = 'Data/test' #directorio de prueba
#train_dir = 'Data/minitrain' #directorio de entrenamiento mini
#test_dir = 'Data/minitest' #directorio de prueba mini

num_class = 40 #cuantas clases
epochs = 5 #cuantas veces entrenar. En cada epoch hace una mejora en los parametros
batch_size = 25 #batch para hacer cada entrenamiento. Lee 50 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria
num_train = 202599  #numero de imagenes en train
num_test = 7864 #numero de imagenes en test
#num_train = 3654  #numero de imagenes en minitrain
#num_test = 4210 #numero de imagenes en minitest

test_steps = num_test // batch_size
epoch_steps = num_train // batch_size

#Generamos las imagenes
gentrain = ImageDataGenerator(rescale=1. / 255.) #indica que reescale cada canal con valor entre 0 y 1.
train = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')
gentest = ImageDataGenerator(rescale=1. / 255)
test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='binary')
exit()

#Capas de la red neuronal
model=Sequential()
#Primera capa: Convolucional.
model.add(Conv2D(10, (3,3), input_shape=(ih, iw,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Segunda capa Convolucional
model.add(Conv2D(20, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Flatten
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#Dropout
model.add(Dropout(0.2))
#Clasificador
model.add(Dense(40))
model.add(Activation('sigmoid'))
model.add(Flatten())

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#exit()

log_dir="logs/fit2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#python -m tensorboard.main --logdir=/Graph

print("Logs:")
print(log_dir)
print("__________")

#exit()

model.fit(train, steps_per_epoch=epoch_steps, epochs=epochs, validation_data=test, validation_steps=test_steps, callbacks=[tbCallBack])
