import os
import pandas
from setup import Setup, MLOperations
from ml import LogisticRegression, KNearestNeigbour
from eda import Clean

clearCMD = 'cls'
lastAppliedModel = ''

rawTrainData, rawTestData, config = Setup()

cleanTrainData = Clean(rawTrainData.copy())
cleanTestData = Clean(rawTestData.copy())

predictionData = pandas.DataFrame(columns=cleanTestData.columns)

parameters = {
    'Input Parameters' : ['Age', 'H', 'W'],
    'Prediction Element' : 'Obesity_Level'
}

userInput = MLOperations()
while userInput != "E":
    os.system(clearCMD)
    if userInput == "2":
        print("Input Training Data")
        print(cleanTrainData.to_string(), "\n")
    if userInput == "3":
        print("Input Test Data")
        print(cleanTestData.to_string(), "\n")
    if userInput == "4":
        lastAppliedModel = LogisticRegression(predictionData, cleanTrainData, cleanTestData, parameters, config)
    if userInput == "5":
        lastAppliedModel = KNearestNeigbour(predictionData, cleanTrainData, cleanTestData, parameters, config)
    if userInput == "6":
        # PrintPredictionResults(lastAppliedModel, testedSurvivors, testedNonSurvivors)
        pass
    userInput = MLOperations()