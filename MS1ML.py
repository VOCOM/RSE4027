import os
import pandas
from setup import Setup, MLOperations, LogisticRegression, KNearestNeigbour, PrintPredictionResults
from EDA.eda import Clean, Extract

clearCMD = 'cls'
lastAppliedModel = ''

rawTrainData, rawTestData = Setup()

Clean(rawTrainData)
Clean(rawTestData)

extractedTrainData = Extract(rawTrainData)
extractedTestData = Extract(rawTestData)

testedSurvivors = pandas.DataFrame(columns=extractedTestData.columns)
testedNonSurvivors = pandas.DataFrame(columns=extractedTestData.columns)

userInput = MLOperations()
while userInput != "E":
    os.system(clearCMD)
    if userInput == "2":
        print("Input Training Data")
        print(extractedTrainData.to_string(), "\n")
    if userInput == "3":
        print("Input Test Data")
        print(extractedTestData.to_string(), "\n")
    if userInput == "4":
        lastAppliedModel, testedSurvivors, testedNonSurvivors = LogisticRegression(lastAppliedModel, testedSurvivors, testedNonSurvivors, extractedTrainData, extractedTestData)
    if userInput == "5":
        lastAppliedModel, testedSurvivors, testedNonSurvivors = KNearestNeigbour(lastAppliedModel, testedSurvivors, testedNonSurvivors, extractedTrainData, extractedTestData)
    if userInput == "6":
        PrintPredictionResults(lastAppliedModel, testedSurvivors, testedNonSurvivors)
    
    userInput = MLOperations()