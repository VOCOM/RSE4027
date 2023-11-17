import os
import pandas
from setup import Setup, MLOperations
from ml import LogisticRegression, KNearestNeigbour, PredictionResults
from eda import Clean

lastAppliedModel = ''

rawTrainData, rawTestData, config = Setup()

clearCMD = config['Clear Command']

cleanTrainData = Clean(rawTrainData.copy())
cleanTestData = Clean(rawTestData.copy())

predictionData = pandas.DataFrame(columns=cleanTestData.columns)

parameters = {
    'Input Parameters' : ['Age', 'H', 'W', 'GR', 'FAVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE'],
    'Prediction Element' : 'Obesity_Level'
}

userInput = 0
while userInput != "E":
    os.system(clearCMD)
    if userInput == "2":
        print("Input Training Data")
        print(cleanTrainData.to_string(), "\n")
    if userInput == "3":
        print("Input Test Data")
        print(cleanTestData.to_string(), "\n")
    if userInput == "4":
        lastAppliedModel, predictionData = LogisticRegression(predictionData, cleanTrainData, cleanTestData, parameters, config)
    if userInput == "5":
        lastAppliedModel, predictionData = KNearestNeigbour(predictionData, cleanTrainData, cleanTestData, parameters, config)
    if userInput == "6":
        PredictionResults(lastAppliedModel, predictionData, 'Obese', config)
    userInput = MLOperations()