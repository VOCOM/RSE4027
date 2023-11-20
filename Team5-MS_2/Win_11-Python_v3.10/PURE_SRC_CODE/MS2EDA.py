## 
# Changelog:
# 
##

import os
from setup import Setup, EDAOperations, Plots
from eda import Clean, DropAbnormalities, NaEntries, CorrelationMatrix, ObeseProbability

clearCMD = 'cls'
rawTrainData, rawTestData, config = Setup()
# Classifications
classification = {
    'Insufficient_Weight' : 0,
    'Normal_Weight' : 1,
    'Overweight_Level_I' : 2,
    'Overweight_Level_II' : 3,
    'Obesity_Type_I' : 4,
    'Obesity_Type_II' : 5,
    'Obesity_Type_III' : 6
}
# Binary Discretisation
binary = {
    'no' : 0,
    'yes' : 1
}

config.update({'Classifications' : classification})
config.update({'No. of Classes' : len(classification)})
config.update({'Binary' : binary})

cleanTrainData = Clean(rawTrainData.copy(), config)
cleanTestData = Clean(rawTestData.copy(), config)

userInput = EDAOperations()

while userInput != "E":
    os.system(clearCMD)
    if userInput == "2":
        print(rawTrainData.to_string(), '\n')
    if userInput == "3":
        print(cleanTrainData.to_string(), '\n')
    if userInput == "4":
        print(rawTestData.to_string(), '\n')
    if userInput == "5":
        print(cleanTestData.to_string(), '\n')
    if userInput == "6":
        NaEntries(cleanTrainData)
    if userInput == "7":
        CorrelationMatrix(cleanTrainData)
    if userInput == "8":
        ObeseProbability(cleanTrainData)
    userInput = EDAOperations()