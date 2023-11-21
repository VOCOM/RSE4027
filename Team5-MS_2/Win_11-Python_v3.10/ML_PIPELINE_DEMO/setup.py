## 
# @author Muhammad Syamim Bin Shamsulbani
# @brief Machine learning & Artificial Intelligence Program for RSE4207
# @version 1.0
# @date 16/11/23
#
# Changelog:
# - 16/11/23
#   Setup returns configuration parameters
##

# OS Import
import os

# Dataframe Import
import pandas

# Graphing Imports
import matplotlib.pyplot as plt
import seaborn as sns

# Math Import
import numpy

def Setup(configOnly = False):
    configFile = open('../SCRIPTS_CFG/config.txt')
    clearCMD = ''
    isUnified = False
    splitRatio = 0
    maxIterations = 0
    kMeans = 0
    mClass = False
    estimators = 0
    savePath = ''
    resultsPath = ''
    for line in configFile.readlines():
        if "Clear Command: " in line:
            clearCMD = line.strip().split("\"")[1]
        if "Unified dataset: " in line:
            isUnified = bool(line.strip().split("\"")[1].lower() == "true")
        if "SplitRatio (Train/Test): " in line:
            if isUnified:
                line = line.strip().split("\"")[1].split("/")
                splitRatio = float(line[0]) /100
        if "Training dataset: " in line:
            trainingDataPath = line.strip().split("\"")[1].replace("\\","/")
        if "Testing dataset: " in line:
            testDataPath = line.strip().split("\"")[1].replace("\\","/")
        if "Max Iteration: " in line:
            maxIterations = int(line.strip().split("\"")[1])
        if "K-Means: " in line:
            kMeans = int(line.strip().split("\"")[1])
        if "Multi-Class: " in line:
            mClass = bool(line.strip().split("\"")[1].lower() == "true")
        if "Estimators: " in line:
            estimators = int(line.strip().split("\"")[1])
        if "Save Location: " in line:
            savePath = line.strip().split("\"")[1]
        if "Results: " in line:
            resultsPath = line.strip().split("\"")[1]

    if not configOnly:
        trainData = pandas.read_csv(trainingDataPath)
        trainData.dropna(axis='index', inplace=True, how='all')
        if isUnified:
            testData = trainData.copy()
            maxIndex = len(trainData)
            splitIndex = int(maxIndex * splitRatio)
            i = 0
            while i < splitIndex:
                testData.drop(axis='index', labels=i, inplace=True)
                i += 1
            while i < maxIndex:
                trainData.drop(axis='index', labels=i, inplace=True)
                i += 1
            testData.reset_index(inplace=True)
            trainData.reset_index(inplace=True)
            testData.drop(axis='columns', labels='index', inplace=True)
            trainData.drop(axis='columns', labels='index', inplace=True)
        else:
            testData = pandas.read_csv(testDataPath)
    
    config = {
        'Clear Command' : clearCMD,
        'Unified' : isUnified,
        'Split Ratio' : splitRatio,
        'Multi-Class' : mClass,
        'Max Iteration' : maxIterations,
        'K-Means' : kMeans,
        'Estimators' : estimators,
        'Save Path' : savePath,
        'Result Path' : resultsPath
    }

    if configOnly:
        return config
    else:
        return trainData, testData, config

def EDAOperations():
    userInput = ''
    operationList = [
        " ",
        "1) Clear Screen",
        "2) Print original data table",
        "3) Print cleaned data table",
        "4) Print original test table",
        "5) Print cleaned test table",
        "6) Print information about missing entries",
        "7) Correlation Matrix",
        "8) Specific data column vs obese probability",
        "E) Exit Program"
    ]

    for operation in operationList:
        print(operation)
    userInput = input("Operation:").capitalize()

    return userInput

def JOperations_s():
    userInput = ''
    operationList = [
        "1) Clear Screen",
        "2) Print extracted data table",
        "3) Print extracted test table",
        "4) Logistic Regression",
        "5) K-Nearest Neighbour",
        "6) Random Forest",
        "7) Save Metrics",
        "8) Save Results",
        "E) Exit Program"
    ]

    for operation in operationList:
        print(operation)
    
def JOperations_e():
    userInput = input("Operation:").capitalize()

    return userInput

def MLOperations():
    userInput = ''
    operationList = [
        "1) Clear Screen",
        "2) Print extracted data table",
        "3) Print extracted test table",
        "4) Logistic Regression",
        "5) K-Nearest Neighbour",
        "6) Random Forest",
        "7) Save Metrics",
        "8) Save Results",
        "E) Exit Program"
    ]

    for operation in operationList:
        print(operation)
    userInput = input("Operation:").capitalize()

    return userInput

def FilteredTable():
    header = list(extractedData.keys())
    print(header)
    userInput = input("Choose a category to search for: ")
    if userInput in header:
        print(extractedData[userInput].to_string())
    else:
        os.system(clearCMD)
        print("Category not found\n")
