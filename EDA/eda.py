from FileReader.DataTypes import DATA

def FindAll(data, categoryPos, category):
    for dataLine in data.data:
        if dataLine[categoryPos] == category:
            print(dataLine)
