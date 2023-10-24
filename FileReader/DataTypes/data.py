from tabulate import tabulate

class DATA:
    def __init__(self, header, namePos = -1) -> None:
        self.header = header
        self.data = []
        self.dict = {}
        self.namePos = namePos
        for header in self.header:
            self.dict.update({header:[]})

    def AppendData(self, dataLine):
        self.data.append(dataLine)

    def CleanData(self):
        for dataLine in self.data:
            if self.namePos > -1:
                self.CleanName(dataLine)
                self.CleanAge(dataLine)
                self.CleanNumSiblingSpouse(dataLine)
                self.CleanNumParentChild(dataLine)
                self.CleanSurvived(dataLine)

    def CleanName(self, dataLine):
        dataLine[self.namePos] = dataLine[self.namePos].replace("\"", "")

    def CleanAge(self, dataLine):
        dataLine[7] = int(dataLine[7].split('.')[0])

    def CleanNumSiblingSpouse(self, dataLine):
        dataLine[9] = int(dataLine[9])

    def CleanNumParentChild(self, dataLine):
        dataLine[10] = int(dataLine[10])

    def CleanSurvived(self, dataLine):
        if dataLine[11].upper() == "YES":
            dataLine[11] = 1
        elif dataLine[11].upper() == "NO":
            dataLine[11] = 0

    def GenerateDictionary(self):
        headerIndex = 0
        for header in self.header:
            for data in self.data:
                self.dict.get(header).append(data[headerIndex])
            headerIndex += 1

    def PrintData(self):
        print(tabulate(self.data, headers=self.header))