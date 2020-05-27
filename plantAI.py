from enum import Enum
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


# Болезни растений
class plantDisease(Enum):
    Brown_Rust = 0
    Dark_brown_spotting = 1

# Обработка csv файла с целевыми параметрами изображения
def procFile(dataPath: str, label: int):
    # Чтение данных из csv файла в dataset
    dataset = pd.read_csv(dataPath, sep=',', header=None)

    # Добавление лейбла каждому изображению (к какому классу болезней относится)
    dataset[36] = label

    return dataset

# Объединение датасетов различных групп болезней в один
def uniDataset():
    # Чтение датасетов и присваение лэйблов (классов болезней)
    ds_0 = procFile("data\Brown_rust\\res.csv", 0)
    ds_1 = procFile("data\Dark_brown_spotting\\res.csv", 1)
    ds_2 = procFile("data\Powdery_mildew\\res.csv", 2)
    ds_3 = procFile("data\Pyrenophorosis\\res.csv", 3)
    ds_4 = procFile("data\Root_rot\\res.csv", 4)
    ds_5 = procFile("data\Septoria\\res.csv", 5)
    ds_6 = procFile("data\Smut\\res.csv", 6)
    ds_7 = procFile("data\Snow_mold\\res.csv", 7)
    ds_8 = procFile("data\Striped_mosaic\\res.csv", 8)
    ds_9 = procFile("data\Yellow_rust\\res.csv", 9)

    # Объединение признаков с метками в один датасет
    commonDS = pd.concat([ds_0, ds_1, ds_2, ds_3, ds_4, ds_5, ds_6, ds_7, ds_8, ds_9], axis=0)
    commonDS = commonDS.replace({"-nan(ind)": 0.5})  # Замена непоределенных значений на среднее, корректных вычислений

    return commonDS

# Разделение данных на обучающую и тестовую выборки
def splitDataset(dataset: pd.DataFrame):
    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset[36]

    feature = np.asarray(feature).astype(np.float32)  # Преобразуем все метки в тип float для подачи в нейронную сеть

    # Деление данных на обучающие и тестовые
    trainSize = 0.9
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize,
                                                                            test_size=testSize, random_state=7)

    return feature_train, feature_test, label_train, label_test

# Обучение и тестирование нейронной сети, возвращение обученной модели
def aiTrainTest(feature_train: pd.DataFrame, feature_test: pd.DataFrame, label_train: pd.DataFrame, label_test: pd.DataFrame):
    # Инициализируем нейронную сеть
    model = Sequential()

    # Добавляем и настраиваем слои
    model.add(Flatten(input_shape=(36, )))  # Создаем входной слой равный размерности изображения
    model.add(Dense(42, activation="relu"))  # Добавляем скрытый слой
    model.add(Dense(10, activation="softmax"))  # Добавляем выходной слой на 10 нейронов = количеству типов болезней [0-9]

    # Компилируем модель
    model.compile(optimizer="nadam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Обучаем модель
    model.fit(feature_train, label_train, epochs=3000)

    # Тестирование модели методами keras и вывод данных по точности
    trLoss, trAccuracy = model.evaluate(feature_train, label_train)
    print("Точность на ТРЕНИРОВАЧНЫХ данных = ", trAccuracy)

    testLoss, testAccuracy = model.evaluate(feature_test, label_test)
    print("Точность на ТЕСТОВЫХ данных = ", testAccuracy)

    randLoss, randAccuracy = model.evaluate(feature_test, label_test)
    print("Точность на СЛУЧАНОЙ выборке = ", randAccuracy)

    return model

# Определение болезни по текстурным данным
def findDisease(model: Sequential, textureData: pd.DataFrame):
    # Вывод поданных текстурных данных изображения
    print("Текстурные данные изображения: ", textureData)

    # Получение предсказаний нейронной сети
    modelPredict = model.predict(textureData)

    # Получение номера болезни из предсказаний нейронной сети
    numDisease = np.argmax(modelPredict)
    print("Номер болезни: ", numDisease)
    print("Название болезни: ", plantDisease(numDisease))

# Подготовка программы к работе
def preparePlantAI():
    # Читаем и объединяем текстурные данные нормализованных изображений
    dataset = uniDataset()

    # Подготавливаем и разделяем данные для обучения
    dataSplit = splitDataset(dataset)

    # Обучаем и тестируем нейронную сеть
    model = aiTrainTest(dataSplit[0], dataSplit[1], dataSplit[2], dataSplit[3])

    return model


# Тестирование работы программы
AI_MODEL = preparePlantAI() # Получение обученной нейронной сети, вывод тестировачных данных

dataset = uniDataset() # Получение датасета готовых реальных текстурных данных
dataset = dataset.iloc[:, :-1] #  Удаление добавленных лэйблов
dataset = np.asarray(dataset).astype(np.float32)  # Преобразование данных в float
dataset = pd.DataFrame(dataset)  # Преобразование в pandas.DataFrame для удобства

# Создание тестировачных сетов реальных текстурных данных
testDS1 = dataset.loc[[3]]
testDS2 = dataset.loc[[14]]
testDS3 = dataset.loc[[22]]

# Тестирования работы программы, выдача названия болезни по текстурным данным
findDisease(AI_MODEL, testDS1)
findDisease(AI_MODEL, testDS2)
findDisease(AI_MODEL, testDS3)
