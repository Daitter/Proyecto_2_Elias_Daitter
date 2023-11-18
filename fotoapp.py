import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFilter

#Declaro donde se van a guardar las imagenes para que no se sobreescriban las imagenes originales y asi es mas facil guardarlas
folder='images/imagen_guardada.jpg'

def redimensionar_imagen(ruta, palabra):
    # Definir dimensiones recomendadas para las redes sociales
    dimensiones = {
        "Youtube": (1280, 720),
        "youtube": (1280, 720),
        "YOUTUBE": (1280, 720),

        "Instagram": (1080, 1080),
        "instagram": (1080, 1080),
        "INSTAGRAM": (1080, 1080),

        "Twitter": (1024, 512),
        "twitter": (1024, 512),
        "TWITTER": (1024, 512),

        "Facebook": (1200, 628),
        "facebook": (1200, 628),
        "FACEBOOK": (1200, 628)
    }

    # Abrir la imagen con open cv
    imagen =  cv2.imread(ruta, cv2.IMREAD_COLOR)

    # Obtener las dimension recomendada segun la red social y redimensionarla
    if palabra in dimensiones:
      imagen_redimensionada = cv2.resize(imagen, dimensiones[palabra])

    # Muestro imagen - Use pyplot porque cv2 imshow me daba problemas
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(imagen_redimensionada, cv2.COLOR_BGR2RGB))

def contraste(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

    # Calcular el histograma
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # Calcular la función de distribución acumulativa
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Ecualizar la imagen
    img_ecualizada = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    img_ecualizada = img_ecualizada.reshape(img.shape)

    # Mostrar las imágenes original y ecualizada en una figura
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen Original')

    plt.subplot(1, 2, 2)
    plt.imshow(img_ecualizada, cmap='gray')
    plt.title('Imagen Ecualizada')
    
    #Guardamos las imagenes
    cv2.imwrite(folder, img)

def filtros(ruta,filtro):
  # Abrimos la imagen
  img=Image.open(ruta)

  #Muestro la imagen original y la que tiene el filtro indicado
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(img)
  plt.title('Imagen Original')

  plt.subplot(1, 2, 2)
  # Filtro BLUR
  if filtro=='BLUR':
    img_blur=img.filter(ImageFilter.BLUR)
    plt.imshow(img_blur)
    plt.title('Imagen Blur')
  # Filtro CONTOUR
  if filtro=='CONTOUR':
    img_contour=img.filter(ImageFilter.CONTOUR)
    plt.imshow(img_contour)
    plt.title('Imagen Contour')
  # Filtro DETAIL
  if filtro=='DETAIL':
    img_detail=img.filter(ImageFilter.DETAIL)
    plt.imshow(img_detail)
    plt.title('Imagen Detail')
  # Filtro EDGE ENHANCE
  if filtro=='EDGE_ENHANCE':
    img_edge_enhace=img.filter(ImageFilter.EDGE_ENHANCE)
    plt.imshow(img_edge_enhace)
    plt.title('Imagen Enhance')
  # Filtro EDGE ENHANCE MORE
  if filtro=='EDGE_ENHANCE_MORE':
    img_enhance_more=img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    plt.imshow(img_enhance_more)
    plt.title('Imagen Enhance more')
  # Filtro EMBOSS
  if filtro=='EMBOSS':
    img_emboss=img.filter(ImageFilter.EMBOSS)
    plt.imshow(img_emboss)
    plt.title('Imagen Emboss')
  # Filtro FIND EDGES
  if filtro=='FIND_EDGES':
    img_find_edges=img.filter(ImageFilter.FIND_EDGES)
    plt.imshow(img_find_edges)
    plt.title('Imagen Find Edges')
  # Filtro SHARPEN
  if filtro=='SHARPEN':
    img_sharpen=img.filter(ImageFilter.SHARPEN)
    plt.imshow(img_sharpen)
    plt.title('Imagen Sharpen')
  # Filtro SMOOTH
  if filtro=='SMOOTH':
    img_smooth=img.filter(ImageFilter.SMOOTH)
    plt.imshow(img_smooth)
    plt.title('Imagen Smooth')

  fig, axs = plt.subplots(3, 3, figsize=(10, 10))
  axs[0, 0].imshow(img)
  axs[0, 0].set_title("Original", color='black')

  cont=0
  # Un bucle for para aplicar todos los filtros y mostrarlos
  for i in todosLosFiltros:
    imagen_filtrada = img.filter(i)
    axs[cont//3, cont%3].axis('off')
    axs[cont//3, cont%3].imshow(imagen_filtrada)
    fil=limpiarStr(str(i))
    axs[cont//3, cont%3].set_title(fil)
    cont+=1
    
# Esta funcion la uso para que quede escrito de una mejor manera
def limpiarStr(string):
  string=string.replace('<class','')
  string=string.replace('PIL.ImageFilter.','')
  string=string.replace('>','')
  return(string)    

#Hice una lista con todos los filtros que iba a usar para que sea mas facil aplicarlos todos en el bucle for
todosLosFiltros = [ImageFilter.BLUR,
                   ImageFilter.CONTOUR,
                   ImageFilter.DETAIL,
                   ImageFilter.EDGE_ENHANCE,
                   ImageFilter.EDGE_ENHANCE_MORE,
                   ImageFilter.EMBOSS,
                   ImageFilter.FIND_EDGES,
                   ImageFilter.SHARPEN,
                   ImageFilter.SMOOTH]

def imagen_binarizada(ruta):
  # Abrimos la imagen
  img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
  # Detecto los bordes de la imagen
  boceto = cv2.Canny(img, 100, 300)
  # Muestro la imagen original y la binarizada
  plt.subplot(121)
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.title('Imagen Original')
  plt.subplot(122)
  plt.imshow(boceto, cmap='gray')
  plt.title('Imagen Binarizada')
  plt.show()
    
# Yo creo que mi funcion de imagen binarizada es una buena herramienta para pintores y dibujantes
# gracias a la funcion Canny de cv2 que resalta muy bien los bordes, entonces les facilita el trabajo a los mismos
# En conclusion, funciona bien y es facil de usar :3