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

    def CleanName(self, dataLine):
        dataLine[self.namePos] = dataLine[self.namePos].replace("\"", "")

    def GenerateDictionary(self):
        for header in self.header:
            for data in self.data:
                self.dict.get(header).append(data)

    def PrintData(self):
        print(tabulate(self.data, headers=self.header))