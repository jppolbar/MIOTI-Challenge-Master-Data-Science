import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, io, filters, color
from skimage.color import rgb2gray
from ipywidgets import interactive, IntSlider, FloatSlider, VBox, HBox
from IPython.core.display import display, HTML
from skimage.feature import corner_peaks
from skimage.feature import corner_moravec

import math
import warnings
warnings.filterwarnings("ignore")

from skimage.transform import warp, EuclideanTransform, AffineTransform

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

    for point in points:
        out[circle(point[1], point[0], 10)] = [1, 0, 0]
    
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