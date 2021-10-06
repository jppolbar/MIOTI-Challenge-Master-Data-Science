# Helpers para OCR
# Diego García Morate
# diegogm@faculty.mioti.es

from skimage import data, io, filters, transform, img_as_ubyte
from skimage.util import invert
import os
import shutil
from skimage.measure import label
from skimage.transform import rotate
from skimage.color import rgb2gray
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import threshold_otsu
import numpy as np
import glob
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def mostrar_frames(video):
    """Muestra una secuencia de frames almacenada en una lista"""
    if len(video) > 0:
        if video[0].ndim == 2:
            t = type(video[0][0][0].item())
            if t == float:
                for frame in video:
                    io.imshow(frame, cmap='gray')
                    io.show()
            elif t == int:
                for frame in video:
                    io.imshow(frame, cmap='gray')
                    io.show()
            elif t == bool:
                for frame in video:
                    io.imshow(frame.astype(float), cmap=plt.cm.gray)
                    io.show()
            else:
                for frame in video:
                    io.imshow(frame)
                    io.show()        
        else:
            for frame in video:
                io.imshow(frame)
                io.show()

def mostrar_blobs(img, blobs):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    
    for blob in blobs:    
        y, x, r = blob
        c = plt.Circle((x, y), r, color='#FFFF00AA', linewidth=2, fill=False)
        ax.add_artist(c)
        
def mostrar_rois(img, rois):
    """Muestra los ROIS en una imagen dada"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    for roi in rois:
        top, left, bottom, right = roi.bbox
        rect = mpatches.Rectangle((left, top), right - left, bottom - top,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()        

def extraer_ventana(img, x, y, width, height):
    """Función que extrae una ventana de una imagen dada"""
    if len(img.shape) == 3: # RGB
        return img[y: y + height, x : x+width, :]
    elif len(img.shape) == 2: # Monocromo o escala de grises
        return img[y: y + height, x : x+width]

def extraer_limites(img):
    """ Obtiene los límites de una imagen monocromo"""

    suma_y = invert(img).sum(axis=0)
    suma_x = invert(img).sum(axis=1)
    
    width = img.shape[0]
    height = img.shape[1]

    left = 0
    right = width - 1 
    top = 0
    bottom = height - 1

    # Recorremos de izquierda a derecha
    for col in range(width):
        if suma_y[col] == 0:
            left = col
        else:
            break

    # Derecha a izquierda
    for col in range(width - 1, -1, -1):
        if suma_y[col] == 0:
            right = col
        else:
            break

    # Recorremos de arriba a abajo
    for col in range(height):
        if suma_x[col] == 0:
            top = col
        else:
            break

    # abajo a arriba
    for col in range(height - 1, -1, -1):
        if suma_x[col] == 0:
            bottom = col
        else:
            break

    #print("left: {}  right: {}  top: {}  bottom: {}".format(left, right, top, bottom))
    return left, right, top, bottom


def genera_dataset(corpus_folder, fonts_folder="fonts"):
    try:
        shutil.rmtree(corpus_folder)
    except:
        pass
    
    try:
        os.mkdir(corpus_folder)
    except:
        pass
    
    font_sizes = [16, 24, 32]
    angles = [-30, -15, 0, 15, 30]
    chars = list('abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZ0123456789')
    
    num_instance = 0
    for font_name in glob.glob(fonts_folder + "/*.tt*"):
        for font_size in font_sizes:
            font = ImageFont.truetype(font_name, size=font_size)
            
            for angle in angles:
                index_char = 0
                for char in chars:
                    # Pinto la letra en una imagen
                    img = Image.new('RGB', (32, 32), color='white')
                    canvas = ImageDraw.Draw(img)
                    canvas.text((0,0), char, fill='rgb(0,0,0)', font=font)

                    # La convierto en formato sklearn
                    img_letra = np.asarray(img)

                    # Conversión a gris
                    img_letra_gris = rgb2gray(img_letra)

                    # Conversión a monocromo
                    umbral_otsu = threshold_otsu(img_letra_gris)
                    img_letra_mono = img_letra_gris > umbral_otsu

                    # Rotamos imagen
                    img_letra_mono = rotate(img_letra_mono, angle, resize=False)

                    #io.imshow(img_letra_mono)
                    #io.show()

                    # Extraemos los limites
                    left, right, top, bottom = extraer_limites(img_letra_mono)
                    ventana = extraer_ventana(img_letra_gris, left, top, right - left, bottom - top)

                    #io.imshow(ventana)
                    #io.show()

                    ventana_ub = img_as_ubyte(ventana)
                    io.imsave("corpus/{}_{}_{}.png".format(char, index_char, num_instance), ventana_ub)
                    index_char += 1
                    num_instance += 1
                    
def carga_dataset(folder):
    target_width = 16 # mio
    target_height = 16 # mio


    num_instances = len(glob.glob(folder + "/*.png"))
    num_attributes = target_width * target_height

    X = np.zeros((num_instances, num_attributes))
    y = []
    clases_dict = {}

    index = 0
    for f in sorted(glob.glob(folder + "/*.png")):
        img = io.imread(f)
        
        # Reescalamos la letra a nuestro tamaño objetivo
        img_scaled = transform.resize(img, (target_width, target_height))

        # Almacenamos en X la letra actual
        X[index] = img_scaled.reshape(target_width*target_height)

        # Obtenemos la letra y el indice a la que pertenece
        filename, ext = os.path.splitext(os.path.basename(f))
        letter, idx, idx_global = filename.split("_")
        
        clases_dict[int(idx)] = letter
        y.append(int(idx))

        index += 1
        
    num_clases = len(set(y))
    print(clases_dict)
    
    print("Cargadas {} instancias con {} atributos. Clases {}.".format(num_instances, num_attributes, num_clases))
    return X, np.array(y), clases_dict


def dibujar_matches(img, matches, show_original_img=True):
    img_pil = Image.fromarray(img)

    canvas = ImageDraw.Draw(img_pil)
    if not show_original_img:
        canvas.rectangle([(0,0), img_pil.size], fill='rgb(255,255,255)')

    font = ImageFont.truetype("fonts/arial.ttf", size=64)
    for match in matches:
        roi, letra = match
        top, left, bottom, right = roi.bbox

        canvas.text((left, top), letra, fill='rgb(0,0,255)', font=font)
    
    display(img_pil)

    