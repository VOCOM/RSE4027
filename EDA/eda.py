import os

def loadData(dataPath):
    trainPath = dataPath + "/train"
    testPath = dataPath + "/test"
    print("Reading training data files")
    for filePath in os.listdir(trainPath):
        loadCSV(os.path.join(trainPath,filePath))
    print("Reading testing data files")
    for filePath in os.listdir(testPath):
        loadCSV(os.path.join(testPath,filePath))

def loadCSV(filePath):
    print("Reading CSV file from " + filePath)
    file = open(filePath, encoding="utf-8-sig")

    print("Extracting header data")
    line = file.readline()
    lineTuple = line.strip().split(",")
    print(lineTuple)
    
    file.close()