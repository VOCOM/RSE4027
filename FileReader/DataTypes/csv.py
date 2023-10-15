from tabulate import tabulate

class CSV:
    def __init__(self, file) -> None:
        self.header = []
        self.headerSize = 0
        self.dataSet = []
        self.ExtractHeader(file)
        self.ExtractData(file)
    
    def ExtractHeader(self,file):
        line = file.readline()
        self.header = line.strip().split(",")
        self.headerSize = len(self.header)

    def ExtractData(self, file):
        index = 0
        for line in file:
            nameStart = line.find("\"")
            nameEnd = line.rfind("\"") + 1
            name = line[nameStart:nameEnd]
            data = line[:nameStart].strip(",").split(",")
            data.append(name.replace("\"",""))
            for subData in line[nameEnd::].strip(",").strip("\n").split(","):
                data.append(subData)
            if len(data) != self.headerSize:
                print("There is missing information on line " + str(index) + "")
            else:
                self.dataSet.append(data)
            index += 1

    def PrintData(self):
        print(tabulate(self.dataSet, headers=self.header))
