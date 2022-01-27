# mioti.cnn.cv : Red Neuronal Convolucional MIOTI - ComputerVision

from os.path import join

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19

from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet

from tensorflow.keras import layers

def creaModeloConPesosImagenetParaEntrenar (nombre_modelo, num_clases):
    
    modelo_nuevo = None
    
    if nombre_modelo == "VGG19":
        
        modelo_nuevo_parcial = VGG19(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = preprocess_input_vgg19(x)
        x = modelo_nuevo_parcial(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_clases, activation='softmax')(x)
        modelo_nuevo = tf.keras.Model(inputs, outputs)
        
    elif nombre_modelo == "EfficientNetB0":
        
        modelo_nuevo_parcial = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = preprocess_input_efficientnet(x)
        x = modelo_nuevo_parcial(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(num_clases, activation='softmax')(x)
        modelo_nuevo = tf.keras.Model(inputs, outputs)
        
    else:
        
        raise Exception("Modelo '" + modelo + "' no soportado")
    
    return modelo_nuevo

def muestraGraficasDelHistoricoDeEntrenamiento (history_data):
    acc = history_data['accuracy']
    val_acc = history_data['val_accuracy']
    loss = history_data['loss']
    val_loss = history_data['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')

    plt.legend()
    plt.show()

class ModeloPractica:
    
    def __init__(self, modelo_base, dir_model=None):
        
        self.modelo_base = modelo_base
        self.model = None
        self.classes = ['Cebolla', 'Manzana', 'Naranja']
        
        ready = False
        
        if modelo_base == "VGG19":
            
            if dir_model == None:
                dir_model = "./newmodel/VGG19-sgd-entrenamientosoloultimascapas-batchsize8/"
            file_weights = join (dir_model, "model.h5")
            file_model = join (dir_model, "model.json")
            ready = True
                
        elif modelo_base == "EfficientNetB0":
            
            if dir_model == None:
                dir_model = "./newmodel/EfficientNetB0-sdg-entrenamientosoloultimascapas-batchsize8/"
            file_weights = join (dir_model, "model.h5")
            file_model = join (dir_model, "model.json")
            ready = True
                
        else:
            raise Exception("Modelo '" + modelo + "' no soportado")
        
        if ready:
            
            try:
                
                json_file = open(file_model, 'r')
                model_json = json_file.read()
                json_file.close()
                self.model = tf.keras.models.model_from_json(model_json)
                self.model.load_weights(file_weights)
                
            except Exception as ex:
                
                print("Error al cargar el modelo: ", modelo_base, " - Excepción: ", ex)
                
    def get_model (self):
        return self.model
        
    def preprocess_input (self, x):
        if self.modelo_base == "VGG19":
            nx = preprocess_input_vgg19(x)
        elif self.modelo_base == "EfficientNetB0":
            nx = preprocess_input_efficientnet(x)
        else:
            nx = x
        return nx
