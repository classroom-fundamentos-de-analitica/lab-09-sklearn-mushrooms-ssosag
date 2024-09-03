import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


def loadData():
    train = pd.read_csv("train_dataset.csv")
    test = pd.read_csv("test_dataset.csv")
    return (train, test)


def splitData(data):
    x = data.drop(["type"], axis=1)
    y = data["type"]
    return (x, y)


def oneHotEncoder(xTrain, xTest):
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output=False, drop="first")

    X_train_encoded = encoder.fit_transform(xTrain)
    X_test_encoded = encoder.transform(xTest)

    return (X_train_encoded, X_test_encoded)


def entrenarModelo(xTrain, yTrain):
    modelo = RandomForestClassifier()
    modelo.fit(xTrain, yTrain)

    return modelo


def saveModel(model):
    import pickle

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    return None


def main():
    train, test = loadData()
    xTrain, yTrain = splitData(train)
    xTest, yTest = splitData(test)
    xTrainEncoded, xTestEncoded = oneHotEncoder(xTrain, xTest)
    modelo = entrenarModelo(xTrainEncoded, yTrain)
    saveModel(modelo)
    yPred = modelo.predict(xTestEncoded)

    print(accuracy_score(yTest, yPred))


main()
