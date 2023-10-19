from tabulate import tabulate

class DATA:
    def __init__(self, header, namePos = -1) -> None:
        # TODO: Change DATA to dictionary for [key : value] pair -> [str : array]
        self.header = header
        self.data = []
        self.dict = {}
        self.namePos = namePos
        for header in self.header:
            self.dict.update({header:[]})

    def AppendData(self, dataLine):
        self.data.append(dataLine)
        index = 0
        while index < len(self.header):
            self.dict.update({self.header[index]:dataLine[index]})
            index += 1

    def CleanData(self):
        for dataLine in self.data:
            if self.namePos > -1:
                self.CleanName(dataLine)

    def CleanName(self, dataLine):
        dataLine[self.namePos] = dataLine[self.namePos].replace("\"", "")

    def PrintData(self):
        print(tabulate(self.data, headers=self.header))

    def PrintDict(self):
        print(self.dict.get(self.header))
        for header in self.dict.keys():
            for value in self.dict.get(header):
                # print(value)
                pass