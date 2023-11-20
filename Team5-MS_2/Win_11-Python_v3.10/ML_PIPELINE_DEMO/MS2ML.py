import os
import pandas
from setup import Setup, MLOperations, JOperations_s, JOperations_e
from ml import LogisticRegression, KNearestNeigbour, ConfusionMatrix, RandomForest
from eda import Clean
from utility import SaveSetup, UpdateSaveData, SaveData, SaveResults

lastAppliedModel = None
metrics = None
userInput = None

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
    'Input Parameters' : ['Age', 'BMI', 'GR', 'FAVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'Female', 'Male'],
    'Prediction Element' : 'Obesity_Level'
}

config.update({'Parameters' : parameters})
config.update({'Classifications' : classification})
config.update({'No. of Classes' : len(classification)})
config.update({'Binary' : binary})
config.update({'Cutoff' : 1})

saveData = SaveSetup(config)

cleanTrainData = Clean(rawTrainData.copy(), config)
cleanTestData = Clean(rawTestData.copy(), config)

predictionData = pandas.DataFrame(columns=cleanTestData.columns)

# JOperations_s()

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
        SaveData(saveData, config['Save Path'])
    if userInput == "8":
        SaveResults(predictionData, config)
    if userInput == "4" or userInput == "5" or userInput == "6":
        saveData = UpdateSaveData(lastAppliedModel, saveData, metrics, config)
        ConfusionMatrix(lastAppliedModel, predictionData, config)
    # userInput = JOperations_e()
    userInput = MLOperations()
