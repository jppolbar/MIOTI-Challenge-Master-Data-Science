import os

# Si queremos evitar los Warnings que provoca la nueva librería de tensorflow si no tenemos
# una GPU disponible, podemos descomentar la siguiente línea:

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io, filters
from skimage import transform

import tensorflow as tf

import cv2

import mioti.cnn.cv as mcv

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 0.7
    fontColor              = (0,255,0)
    lineType               = 2

    print("Cargando modelo EfficientNetB0...")
    modelo = mcv.ModeloPractica("EfficientNetB0")
    print("Capturando imágenes de la cámara:")
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Imagen", cv2.WINDOW_NORMAL)
    try:
        while True:
            # Capturamos el frame
            ret, frame = cam.read()
            
            if not ret:
                raise Exception("Cannot capture frame.")
            
            cols = frame.shape[1]
            fils = frame.shape[0]
            
            if cols > fils:
               # la imagen es más ancha que alta
               ini = int((cols / 2) - (fils / 2))
               end = int(ini + fils)
               frame_new = frame[:,ini:end,:]
            else:
               # la imagen es más alta que ancha
               ini = int((fils / 2) - (cols / 2))
               end = int(ini + cols)
               frame_new = frame[ini:end,:,:]
            
            # Lo convertimos de BGR a RGB
            frame_rgb = cv2.cvtColor(frame_new, cv2.COLOR_BGR2RGB)
             
            img_resized = transform.resize(frame_rgb, (224, 224), preserve_range=True).astype(np.ubyte)
            img_expanded = np.expand_dims(img_resized, axis=0)
            img_preprocessed = modelo.preprocess_input(img_expanded)
            preds = modelo.get_model().predict(img_preprocessed)
            #preds_decoded = decode_predictions(preds, top=1)

            #cv2.putText(frame,preds_decoded[0][0][1],bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            cv2.putText(frame_new,format(preds),bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            # Mostramos la imagen
            cv2.imshow("Imagen", frame_new)
            
            k = cv2.waitKey(1) & 0xFF
            if k==27:
               cv2.destroyAllWindows()
               cam.release()
               break
    except KeyboardInterrupt as ex:
        # Si paramos el bloque cancelamos el stream de vídeo.
        cam.release()
        print("Secuencia de video parada:\n\t{}".format(ex))
