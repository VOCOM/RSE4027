from FileReader.DataTypes import DATA

def Find(data, category):
    distributionTable = {}
    for entry in data.dict.get(category):
        if entry in distributionTable.keys():
            distributionTable[entry] += 1
        else:
            distributionTable[entry] = 1
    return distributionTable

def Info(data):
    for header in data.dict.keys():
        print(header, len(data.dict.get(header)), "entries")
    print()
        
    for header in data.dict.keys():
        print(header)
        distributionTable = Find(data, header)
        PrintDistribution(distributionTable)

def PrintDistribution(distributionTable):
        for distribution in distributionTable.items():
            print(distribution)
        print()