## 
# Changelog:
# - 17/11/23
#   Created machine learning library
#   Fixed Logistic Regression (Metrics turned off)
##

# IO Import
import os

# Utility Import
from utility import Metrics, VisualizeMetrics

# Math Import
import numpy

# Dataframe Import
import pandas

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression as LogisticRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Graphing Import
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def LogisticRegression(predictionData, trainData, testData, parameters, config):
    # Reset Prediction dataframe
    predictionData.drop(predictionData.index, inplace=True)
    # Logistic Regression
    regr = linear_model.LogisticRegression(max_iter=config['Max Iteration'])
    X = trainData[parameters['Input Parameters']]
    y = list(trainData[parameters['Prediction Element']])
    regr = regr.fit(X,y)
    # Prediction
    predictions = regr.predict(testData[parameters['Input Parameters']])
    predictionsProbabilities = regr.predict_proba(testData[parameters['Input Parameters']])
    predictionData = testData.copy()
    predictionData.insert(len(predictionData.columns), 'Prediction', predictions)
    predictionData.drop('Abnormal', axis='columns', inplace=True)
    predictionData.sort_values('Prediction', inplace=True)
    # Metrics
    metrics = Metrics(predictionData[parameters['Prediction Element']].values, predictions, predictionsProbabilities, config['Multi-Class'])
    return 'Logistic Regression', predictionData, metrics

def KNearestNeigbour(predictionData, trainData, testData, parameters, config):
    # Reset Prediction dataframe
    predictionData.drop(predictionData.index, inplace=True)
    # K Nearest Neighbor
    K = config['K-Means']

    X = trainData[config['Parameters']['Input Parameters']].values
    y = list(trainData[config['Parameters']['Prediction Element']])
    knn_model = KNeighborsClassifier(n_neighbors = K)
    knn_model.fit(X, y)

    feature_names = config['Parameters']['Input Parameters']
    X_test = testData[config['Parameters']['Input Parameters']].values

    # Prediction

    X = trainData[config['Parameters']['Input Parameters']].values
    y = list(trainData[config['Parameters']['Prediction Element']])
    knn_model = KNeighborsClassifier(n_neighbors = K)
    knn_model.fit(X, y)

    feature_names = config['Parameters']['Input Parameters']
    X_test = testData[config['Parameters']['Input Parameters']].values

    # Prediction
    predictions = knn_model.predict(X_test)#.round(decimals=0).astype(int))
    # predictionProbabilities = knn_model.predict_proba(X_test)
    predictionData = testData.copy()
    predictionData.insert(len(predictionData.columns), 'Prediction', predictions)
    predictionData.drop('Abnormal', axis='columns', inplace=True)
    # Metrics
    # print("KNN Metrics")
    metrics = Metrics(predictionData, predictions, config)

    return 'K-Nearest Neighbour', predictionData, metrics

def ConfusionMatrix(trueValue, predictedValue, predictionParameter):
    label   = [predictionParameter['Prediction Element'], 'Not ' + predictionParameter['Prediction Element']]
    cm = confusion_matrix(trueValue, predictedValue)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot()
    plt.show()

def PredictionResults(lastAppliedModel, predictionData, predictionParameter, metrics, config):
    clearCMD = config['Clear Command']
    userInput = ''
    options = [
        '1) ' + predictionParameter['Prediction Element'],
        '2) Non-' + predictionParameter['Prediction Element'],
        '3) Confusion Matrix',
        '4) Metrics',
        "E) Exit"
    ]
    if not len(predictionData):
        print("No data detected! Apply a model first.")
        return
    while userInput != "E":
        os.system(clearCMD)
        if userInput == "1":
            print(predictionParameter['Prediction Element'], '\n', predictionData.to_string(), '\n')
        if userInput == "2":
            print('Non ' + predictionParameter['Prediction Element'], '\n', predictionData.to_string(), '\n')
        if userInput == "3":
            ConfusionMatrix(predictionData[predictionParameter['Prediction Element']], predictionData['Prediction'], predictionParameter)
        if userInput == "4":
            print(lastAppliedModel, 'Metrics')
            VisualizeMetrics(metrics)
        print("Model Used:", lastAppliedModel)
        for option in options:
            print(option)
        userInput = input("Predictions to list:").capitalize()
    os.system(clearCMD)

def PredictionPlots(data):
    x = 0
    maxVal = 0
    label = []
    plotList = [
        '1) Fare distribution',
        '2) Ticket distribution',
        '3) Country distribution',
        '4) Age distribution',
        '5) Gender distribution',
        '6) Vertical Dependents distribution',
        '7) Horizontal Dependents distribution',
        'E) Return to menu'
    ]
    for plotOption in plotList:
        print(plotOption)
    plotType = input("Plot Type:").capitalize()
    fig, ax = plt.subplots()
    if plotType == "1":
        pass
    if plotType == "2":
        label = list(data["Ticket Class"].unique())
        count = list(data["Ticket Class"].value_counts())
        ax.bar(label, count)
        plt.xlabel("Ticket Class")
    if plotType == "3":
        pass
    if plotType == "4":
        pass
    if plotType == "5":
        pass
    if plotType == "6":
        pass
    if plotType == "7":
        pass
    if plotType == "8":
        pass
    if plotType == "E" or plotType == "":
        plt.close()
        return plotType
    plt.ylabel('Number of Passengers')
    plt.show()
    plt.close()
    os.system(clearCMD)
    return plotType
