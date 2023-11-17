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
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

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

    print(predictionsProbabilities[0])

    # Metrics
    metrics = Metrics(predictionData[parameters['Prediction Element']].values, predictions, predictionsProbabilities, config['Multi-Class'])

    print("Logistic Regression Metrics")
    VisualizeMetrics(metrics)

    return 'Logistic Regression', predictionData

def KNearestNeigbour(predictionData, trainData, testData, parameters, config):
    # Reset Prediction dataframe
    predictionData.drop(predictionData.index, inplace=True)
    # K Nearest Neighbor
    K = config['K-Means']
    X = trainData[parameters['Input Parameters']].values
    y = list(trainData[parameters['Prediction Element']])
    knn_model = KNeighborsRegressor(n_neighbors = K)
    knn_model.fit(X, y)
    knn_model.feature_names_in_ = parameters
    # Prediction
    predictions = knn_model.predict(testData[parameters['Input Parameters']].values).round(decimals=0).astype(int)
    predictionData = testData.copy()
    predictionData.insert(len(predictionData.columns), 'Prediction', predictions)
    predictionData.drop('Abnormal', axis='columns', inplace=True)
    # Metrics
    # print("KNN Metrics")
    # Metrics(testData[parameters['Prediction Element']], predictionData[parameters['Prediction Element']])

    return 'K-Nearest Neighbour', predictionData

def ConfusionMatrix(trueValue, predictedValue, predictionParameter):
    label   = [predictionParameter, 'Not ' + predictionParameter]
    cm = confusion_matrix(trueValue, predictedValue)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot()
    plt.show()

def PredictionResults(lastAppliedModel, predictionData, predictionParameter, config):
    clearCMD = config['Clear Command']
    userInput = ''
    options = [
        '1) ' + predictionParameter,
        '2) Non-' + predictionParameter,
        "E) Exit"
    ]
    if not len(predictionData):
        print("No data detected! Apply a model first.")
        return
    while userInput != "E":
        os.system(clearCMD)
        if userInput == "1":
            print(predictionParameter, '\n', predictionData.to_string(), '\n')
        if userInput == "2":
            print('Non ' + predictionParameter, '\n', predictionData.to_string(), '\n')
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
