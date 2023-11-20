import os
import pandas
from setup import Setup, MLOperations
from ml import LogisticRegression, KNearestNeigbour, RandomForest, PredictionResults
from eda import Clean

lastAppliedModel = None
metrics = None

rawTrainData, rawTestData, config = Setup()

clearCMD = config['Clear Command']

# Classifications
classification = {
    'Insufficient_Weight' : 0,
    'Normal_Weight' : 1,
    'Overweight_Level_II' : 3,
    'Overweight_Level_I' : 2,
    'Obesity_Type_III' : 6,
    'Obesity_Type_II' : 5,
    'Obesity_Type_I' : 4
}
# Binary Discretisation
binary = {
    'no' : 0,
    'yes' : 1
}
# ML Parameters
parameters = {
    'Input Parameters' : ['Age', 'H', 'W', 'GR', 'FAVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'Female', 'Male'],
    'Prediction Element' : 'Obesity_Level'
}
config.update({'Parameters' : parameters})
config.update({'Classifications' : classification})
config.update({'No. of Classes' : len(classification)})
config.update({'Binary' : binary})
config.update({'Cutoff' : 1})

cleanTrainData = Clean(rawTrainData.copy(), config)
cleanTestData = Clean(rawTestData.copy(), config)

predictionData = pandas.DataFrame(columns=cleanTestData.columns)

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
        lastAppliedModel, predictionData, metrics = LogisticRegression(predictionData, cleanTrainData, cleanTestData, config)
    if userInput == "5":
        lastAppliedModel, predictionData, metrics = KNearestNeigbour(predictionData, cleanTrainData, cleanTestData, config)
    if userInput == "6":
        lastAppliedModel, predictionData, metrics = RandomForest(predictionData, cleanTrainData, cleanTestData, config)
    if userInput == "7":
        PredictionResults(lastAppliedModel, predictionData, metrics, config)
    userInput = MLOperations()