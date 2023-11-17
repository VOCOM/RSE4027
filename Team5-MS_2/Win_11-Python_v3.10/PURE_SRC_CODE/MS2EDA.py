## 
# Changelog:
# 
##

import os
from setup import Setup, EDAOperations, Plots, VisualizeEda
from eda import Clean

clearCMD = 'cls'
rawTrainData, rawTestData, config = Setup()

cleanTrainData = Clean(rawTrainData.copy())
cleanTestData = Clean(rawTestData.copy())

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
    userInput = EDAOperations()