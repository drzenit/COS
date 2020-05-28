from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage import io


# Чтение картинки в память
def readImage(dataPath: str):
    # Чтение изображения
    image = io.imread(dataPath)

    # Вывод прочитанного изображения
    io.imshow(image)
    plt.title("Исходное изображение")
    plt.show()

    return image

# Разбиение изображения на RGB слои
def rgbSplit(image: io.imread):
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    # Вывод слоев изображения
    io.imshow(r)
    plt.title("Красный R")
    plt.show()

    io.imshow(g)
    plt.title("Зеленый G")
    plt.show()

    io.imshow(b)
    plt.title("Синий B")
    plt.show()

    return r, g, b

# Получение текстурных параметров изображения на основе glcm (статистические характеристики)
def getTextureParam(r: io.imread, g: io.imread, b: io.imread):
    # Номрировка слоев RGB
    # r = np.asarray(r) / 255
    # g = np.asarray(g) / 255
    # b = np.asarray(b) / 255

    # Выделение смешанных слоев rgb
    rg = r - g
    rb = r - b
    gb = g - b

    # Возвращение слоев в исходное состояние до нормировки, для glcm матрицы
    # r = np.asarray(r * 255, int)
    # g = np.asarray(g * 255, int)
    # b = np.asarray(b * 255, int)
    # rg = np.asarray(rg * 255, int)
    # rb = np.asarray(rb * 255, int)
    # gb = np.asarray(gb * 255, int)

    # Вывод смешанных слоев изображения
    io.imshow(rg)
    plt.title("Красный-Зеленый RG")
    plt.show()

    io.imshow(rb)
    plt.title("Красно-Синий RB")
    plt.show()

    io.imshow(gb)
    plt.title("Зелено-Синий GB")
    plt.show()

    # Получение различных текстурных признаков
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        # Получение glcm матриц и статистических характеристик для слоев RGB
        glcm = greycomatrix(r, distances=[2], angles=[0], levels=256)
        statsR = greycoprops(glcm, prop)
        glcm = greycomatrix(g, distances=[2], angles=[0], levels=256)
        statsG = greycoprops(glcm, prop)
        glcm = greycomatrix(b, distances=[2], angles=[0], levels=256)
        statsB = greycoprops(glcm, prop)

        # Получение glcm матриц и статистических характеристик для смешанных слоев RGB
        glcm = greycomatrix(rg, distances=[2], angles=[0], levels=256)
        statsRG = greycoprops(glcm, prop)
        glcm = greycomatrix(rb, distances=[2], angles=[0], levels=256)
        statsRB = greycoprops(glcm, prop)
        glcm = greycomatrix(gb, distances=[2], angles=[0], levels=256)
        statsGB = greycoprops(glcm, prop)

        # Перевод данных в pd.DataFrame для удобства
        statsR_DF = pd.DataFrame(statsR)
        statsG_DF = pd.DataFrame(statsG)
        statsB_DF = pd.DataFrame(statsB)
        statsRG_DF = pd.DataFrame(statsRG)
        statsRB_DF = pd.DataFrame(statsRB)
        statsGB_DF = pd.DataFrame(statsGB)

        # Вывод статистических текстурных характеристик
        print("Текстурные данные R для ", prop, ' ', statsR_DF)
        print("Текстурные данные G для ", prop, ' ', statsG_DF)
        print("Текстурные данные B для ", prop, ' ', statsB_DF)
        print("Текстурные данные RG для ", prop, ' ', statsRG_DF)
        print("Текстурные данные RB для ", prop, ' ', statsRB_DF)
        print("Текстурные данные GB для ", prop, ' ', statsGB_DF)


image = readImage("data\Brown_rust\\1.jpg")
r, g, b = rgbSplit(image)
getTextureParam(r, g, b)
