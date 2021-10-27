import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io, filters, color
from skimage.color import rgb2gray
from ipywidgets import interactive, IntSlider, FloatSlider, VBox, HBox
from IPython.core.display import display, HTML
from skimage.feature import corner_peaks
from skimage.feature import corner_moravec
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform, FundamentalMatrixTransform
from skimage.transform import resize
from skimage.transform import warp


import math
import warnings
warnings.filterwarnings("ignore")

from skimage.transform import warp, EuclideanTransform, AffineTransform

def mostrar_frames(video):
    """Muestra una secuencia de frames almacenada en una lista"""
    for frame in video:
        io.imshow(frame)
        io.show()
        
def dibujar_img(img, size=(7,7)):
    fig = plt.figure(figsize=size)
    io.imshow(img)
    io.show()
    
def dibujar_imgs(left, right, size=(10,10)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=size)
    ax[0].imshow(left)
    ax[1].imshow(right)
    io.show()

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


def dibujar_corners(img, corners):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(corners[:, 1], corners[:, 0], '+r', markersize=10)
    plt.show()
    
from skimage.draw import circle

def dibujar_puntos(img, points):
    out = img.copy()
    
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ]
    
    color_idx = 0

    for point in points:
        out[circle(point[1], point[0], 10)] = colors[color_idx]
        color_idx += 1
    
    return out
    
def do_interactive_euclidean():
    def interactive_euclidean(rotation, offset_x, offset_y):
        from skimage.data import chelsea
        img = chelsea()
        transform = EuclideanTransform(
            rotation=rotation, 
            translation=(offset_x, offset_y),
        )
        img_warped = warp(img, transform.inverse, output_shape=img.shape)
        dibujar_img(img_warped, size=(6,4))
        #print(transform.params)

    slider_rotation = FloatSlider(min=-math.pi/2, max=math.pi/2, step=0.01, value=0, continuous_update=False)
    slider_offset_x = IntSlider(min=-300, max=300, step=1, value=0, continuous_update=False)
    slider_offset_y = IntSlider(min=-300, max=300, step=1, value=0, continuous_update=False)

    return interactive(interactive_euclidean, 
                rotation=slider_rotation, 
                offset_x=slider_offset_x, 
                offset_y=slider_offset_y)


def do_interactive_affine():
    def interactive_affine(escala, rotation, offset_x, offset_y, cizalla):
        from skimage.data import chelsea
        img = chelsea()
        transform = AffineTransform(
            scale=(escala, escala), 
            rotation=rotation, 
            translation=(offset_x, offset_y),
            shear=cizalla
        )
        img_warped = warp(img, transform.inverse, output_shape=img.shape)
        dibujar_img(img_warped, size=(4,4))
        #print(transform.params)

    slider_scale = FloatSlider(min=0.5, max=1.5, step=0.05, value=1, continuous_update=False)
    slider_rotation = FloatSlider(min=-math.pi/2, max=math.pi/2, step=0.01, value=0, continuous_update=False)
    slider_cizalla = FloatSlider(min=-math.pi/2, max=math.pi/2, step=0.01, value=0, continuous_update=False)
    slider_offset_x = IntSlider(min=-300, max=300, step=1, value=0, continuous_update=False)
    slider_offset_y = IntSlider(min=-300, max=300, step=1, value=0, continuous_update=False)

    return interactive(interactive_affine, 
                escala=slider_scale, 
                rotation=slider_rotation, 
                offset_x=slider_offset_x, 
                offset_y=slider_offset_y,
                cizalla=slider_cizalla)


def do_interactive_moravec():
    def interactive_corner_peaks(min_distance):
        img = data.moon()
        img_gris = rgb2gray(img)
        coords = corner_peaks(corner_moravec(img_gris), min_distance=min_distance)
        dibujar_corners(img, coords)

    min_distance = IntSlider(min=0, max=50, step=1, value=20, continuous_update=False)
    return interactive(interactive_corner_peaks, min_distance=min_distance)


def extraer_ventana(img, x, y, width, height, debug=False):
    """Función que extrae una ventana de una imagen dada"""
    if debug:
        print("x: {}\ty:{}\twidth:{}\theight:{}".format(x,y,width,height))
    
    if len(img.shape) == 3: # RGB
        return img[y: y + height, x : x+width, :]
    elif len(img.shape) == 2: # Monocromo o escala de grises
        return img[y: y + height, x : x+width]
    else:
        raise TypeException("Número de canales ({}) no soportado.".format(len(img.shape)))
    
def iterador_ventanas(img, max_width=100, max_height=100, debug=False):
    height = img.shape[0]
    width = img.shape[1]
    
    for y in range(0, height - 1, max_height):
        for x in range(0, width - 1, max_width):
            x_ini = x
            x_fin = min(x_ini + max_width - 1, width - 1)
            y_ini = y
            y_fin = min(y_ini + max_height - 1, height - 1)
            
            if debug:
                print("x_ini {:4d}\tx_fin {:4d}\ty_ini {:4d}\ty_fin {:4d}".format(x_ini, x_fin, y_ini, y_fin))
            
            ventana = extraer_ventana(img, x_ini, y_ini, x_fin - x_ini + 1, y_fin - y_ini + 1, debug=False)
            yield ventana

def blend_rgb(img1, img2):
    """Devuelve la mezcla de dos imagenes"""

    if img1.shape != img2.shape:
        raise Exception("Ambas imágenes deben tener el mismo tamaño.")
    
    altura = img1.shape[0]
    anchura = img1.shape[1]

    out = np.array(img1)

    for y in range(altura):
        for x in range(anchura):
            pixel1_zero = (img1[y][x] != [0,0,0]).sum() == 0
            pixel2_zero = (img2[y][x] != [0,0,0]).sum() == 0
            
            if pixel1_zero and pixel2_zero: # Ambos son False
                out[y][x] = [0,0,0]
            elif pixel1_zero == pixel2_zero: # Ambos son iguales y no son ambos false
                #out[y][x] = (img1[y][x] + img2[y][x]) / 2
                out[y][x] = img1[y][x]
            elif pixel1_zero == False:
                out[y][x] = img1[y][x]
            elif pixel2_zero == False:
                out[y][x] = img2[y][x]
    return out

def pega_imagenes(img_a, img_b, debug=False, transform=AffineTransform):
        img_a_gris = rgb2gray(img_a)
        img_b_gris = rgb2gray(img_b)

        descriptor_extractor = ORB(n_keypoints=600, fast_threshold=0.05)

        descriptor_extractor.detect_and_extract(img_a_gris)
        keypoints_a = descriptor_extractor.keypoints
        descriptors_a = descriptor_extractor.descriptors
        print("Extraidos {} keypoints de img_a...".format(len(keypoints_a)))

        descriptor_extractor.detect_and_extract(img_b_gris)
        keypoints_b = descriptor_extractor.keypoints
        descriptors_b = descriptor_extractor.descriptors
        print("Extraidos {} keypoints de img_b...".format(len(keypoints_b)))

        # cross check chequea que sean matches juntos
        matches = match_descriptors(descriptors_a, descriptors_b, cross_check=True)
        print("Localizados {} matches...".format(len(matches)))

        if debug:
            plt.figure(figsize=(20,10))
            plot_matches(plt, img_a, img_b, keypoints_a, keypoints_b, matches, only_matches=True)

        # Invertimos las coordenadas de (y, x) a (x, y)
        src = keypoints_a[matches[:, 0]][:, ::-1]
        dst = keypoints_b[matches[:, 1]][:, ::-1]

        model, inliers = ransac(
            (src, dst),
            transform, 
            min_samples=8,
            residual_threshold=1, 
            max_trials=5000
        )

        if debug:
            plt.figure(figsize=(20,10))
            plot_matches(plt, img_a, img_b, keypoints_a, keypoints_b, matches[inliers], only_matches=True)

        num_inliers = inliers.sum()
        print("Ransac completado con {} inliers...".format(num_inliers))
        
        if debug:
            print("Matriz:\n{}".format(model.params))
        
        img_a_warped = warp(img_a, AffineTransform(), output_shape=(640, 1000))
        img_b_warped = warp(img_b, model, output_shape=(640, 1000))
        
        if debug:
            dibujar_imgs(img_a_warped, img_b_warped)
        
        return blend_rgb(img_a_warped, img_b_warped)