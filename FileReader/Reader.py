import os
from .DataTypes import CSV, DATA

def loadData(dataPath, headerFormat = {}):
    csvSet = []
    for filePath in os.listdir(dataPath):
        csvSet.append(loadCSV(os.path.join(dataPath,filePath), True))

    print("Found", str(len(csvSet)), "Datasets")
    
    compiledData = DATA()
    compiledData.header = csvSet[0].header
    for csv in csvSet:
        for dataLine in csv.dataSet:
            compiledData.data.append(dataLine)
    return compiledData

def loadCSV(filePath, namedFlag = False):
    print("Reading CSV file from " + filePath + "\n")
    file = open(filePath, encoding="utf-8-sig")
    dataSet = CSV(file, namedFlag)
    file.close()
    return dataSet
