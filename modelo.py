import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle


def loadData():
    train = pd.read_csv("train_dataset.csv")
    test = pd.read_csv("test_dataset.csv")
    return (train, test)


def splitData(data):
    x = data.drop(["type"], axis=1)
    y = data["type"]
    return (x, y)


def entrenarModelo(xTrain, yTrain):
    # Crear un pipeline que primero codifica las características y luego entrena el modelo
    pipeline = Pipeline(
        [
            ("encoder", OneHotEncoder(sparse_output=False, drop="first")),
            ("classifier", RandomForestClassifier()),
        ]
    )

    pipeline.fit(xTrain, yTrain)

    return pipeline


def saveModel(model):
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)

    return None


def eval_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def main():
    train, test = loadData()
    xTrain, yTrain = splitData(train)
    xTest, yTest = splitData(test)
    modelo = entrenarModelo(xTrain, yTrain)
    saveModel(modelo)

    yPredTrain = modelo.predict(xTrain)
    yPredTest = modelo.predict(xTest)

    accuracy_train = eval_metrics(yTrain, yPredTrain)
    accuracy_test = eval_metrics(yTest, yPredTest)

    print(f"Train accuracy: {accuracy_train}")
    print(f"Test accuracy: {accuracy_test}")


main()
