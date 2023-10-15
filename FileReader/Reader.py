import os
from .DataTypes import CSV, DATA

def loadData(dataPath):
    csvSet = []
    for filePath in os.listdir(dataPath):
        csvSet.append(loadCSV(os.path.join(dataPath,filePath)))

    print("Found", str(len(csvSet)), "Datasets")
    
    compiledData = DATA()
    compiledData.header = csvSet[0].header
    for csv in csvSet:
        for dataLine in csv.dataSet:
            compiledData.data.append(dataLine)
    return compiledData

def loadCSV(filePath):
    print("Reading CSV file from " + filePath + "\n")
    file = open(filePath, encoding="utf-8-sig")
    dataSet = CSV(file)
    file.close()
    return dataSet
