import os
from .DataTypes import CSV, DATA

def loadData(dataPath, headerFormat = {}):
    csvSet = []
    dataFolder = os.listdir(dataPath)
    for filePath in dataFolder:
        filePath = os.path.join(dataPath,filePath)
        csvSet.append(loadCSV(filePath, True))

    print("Found", str(len(csvSet)), "Datasets")
    
    compiledData = DATA()
    compiledData.header = csvSet[0].header
    for csv in csvSet:
        for dataLine in csv.dataSet:
            # TODO add DATA cleaning here
            compiledData.data.append(dataLine)
    return compiledData

def loadCSV(filePath, namedFlag = False):
    print("Reading CSV file from " + filePath + "\n")
    file = open(filePath, encoding="utf-8-sig")
    dataSet = CSV(file, namedFlag)
    file.close()
    return dataSet
