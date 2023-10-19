import os
from FileReader.DataTypes import DATA

def Find(data, categoryPos, category):
    for dataLine in data.data:
        if dataLine[categoryPos] == category:
            print(dataLine)

def Info(data):
    os.system("clear")
    for header in data.dict.keys():
        print(header, len(data.dict.get(header)), "entries")
    print()