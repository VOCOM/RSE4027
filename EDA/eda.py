from FileReader.DataTypes import DATA

def Find(data, categoryPos, category):
    for dataLine in data.data:
        if dataLine[categoryPos] == category:
            print(dataLine)
