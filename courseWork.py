import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


# Обработка csv файла с целевыми параметрами изображения
def procFile(dataPath: str, label: int):
    # Чтение данных из csv файла в dataset
    dataset = pd.read_csv(dataPath, sep=',', header=None)

    # Добавление лейбла каждому изображению (к какому классу болезней относится)
    dataset[36] = label

    return dataset

# Объединение датасетов различных групп болезней в один
def uniDataset():
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

    commonDS = pd.concat([ds_0, ds_1, ds_2, ds_3, ds_4, ds_5, ds_6, ds_7, ds_8, ds_9], axis=0)
    #commonDS.drop([19], axis='columns', inplace=True)
    #commonDS.drop([25], axis='columns', inplace=True)
    commonDS = commonDS.replace({"-nan(ind)": 0.5})
    return commonDS

# Разделение данных на обучающую и тестовую выборки
def splitDataset(dataset: pd.DataFrame):
    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset[36]

    feature = np.asarray(feature).astype(np.float32)

    # Деление данных на обучающие и тестовые
    trainSize = 0.9
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize,
                                                                            test_size=testSize, random_state=7)

    # Инициализируем нейронную сеть
    model = Sequential()

    # Добавляем и настраиваем слои
    model.add(Flatten(input_shape=(36, )))  # Создаем входной слой равный размерности изображения
    #model.add(Dense(18, activation="sigmoid"))  # Добавляем скрытый слой
    model.add(Dense(42, activation="relu"))  # Добавляем скрытый слой
    #model.add(Dense(16, activation="sigmoid"))  # Добавляем скрытый слой
    #model.add(Dense(12, activation="sigmoid"))  # Добавляем скрытый слой
    model.add(Dense(10, activation="softmax"))  # Добавляем выходной слой на 10 нейронов = количеству типов цифр [0-9]

    # Компилируем модель
    model.compile(optimizer="nadam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Обучаем модель
    model.fit(feature_train, label_train, epochs=3000)

    # Тестирование модели методами keras
    trLoss, trAccuracy = model.evaluate(feature_train, label_train)
    print("Точность TRAIN = ", trAccuracy)
    testLoss, testAccuracy = model.evaluate(feature_test, label_test)
    print("Точность TEST = ", testAccuracy)



splitDataset(uniDataset())
#uniDataset()


