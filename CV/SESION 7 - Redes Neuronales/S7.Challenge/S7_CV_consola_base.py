import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io, filters, transform

from IPython import display

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.applications.inception_v3 import InceptionV3

import warnings
warnings.filterwarnings("ignore")


import cv2

if __name__ == "__main__":
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2

    print("Cargando modelo...")
    # Cargar el modelo de red neuronal convolucional que queráis
    model = InceptionV3(weights='imagenet')

    print("Capturando imágenes de la cámara:")
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL)
    try:
        while True:
            # Capturamos el frame
            ret, frame = cam.read()
            
            if not ret:
                raise Exception("Cannot capture frame.")
            
            # Lo convertimos de BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hacemos preprocesamiento
            img_cropped = frame_rgb [70:370, 150:475]
            img_resized = transform.resize(img_cropped, (299, 299), preserve_range=True).astype(np.ubyte)
            img_expanded = np.expand_dims(img_resized, axis=0)
            img_preprocessed = preprocess_input(img_expanded)
             
            # Preguntad al modelo por la imagen "frame_rgb" y obtener su predicción en "pred_txt"
            preds = model.predict(img_preprocessed)
            print('Predicted: {}'.format(decode_predictions(preds, top=1)))
            preds = decode_predictions(preds, top=1)
            pred_txt = preds[0][0][1]

            cv2.putText(frame,pred_txt,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            # Mostramos la imagen
            cv2.imshow("Imagen", frame)
            
            k = cv2.waitKey(1) & 0xFF
            # Pulsad la tecla "esc" para terminar...
            if k==27:
               cv2.destroyAllWindows()
               cam.release()
               break
    except KeyboardInterrupt as ex:
        # Si paramos el bloque cancelamos el stream de vídeo.
        cam.release()
        print("Secuencia de video parada:\n\t{}".format(ex))
