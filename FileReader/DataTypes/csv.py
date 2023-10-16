class CSV:
    def __init__(self, file, namedFlag = False) -> None:
        self.namedFlag = namedFlag
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
            processedline = line.strip("\n")
            if self.namedFlag:
                processedline = self.ExtractName(processedline)
            if len(processedline) != self.headerSize:
                print("There is missing information on line ", str(index))
            else:
                self.dataSet.append(processedline)
            # print(processedline)
            index += 1

    def ExtractName(self, line):
        nameStart = line.find("\"")
        nameEnd = line.rfind("\"") + 1
        startString = line[:nameStart].strip(",")
        endString = line[nameEnd:].strip(",")
        processedString = []
        for value in startString.split(","):
            processedString.append(value)
        processedString.append(line[nameStart:nameEnd].strip("\"").replace("\"", ""))
        for value in endString.split(","):
            processedString.append(value)
        return processedString


    def PrintData(self):
        print(self.header)
        for dataLine in self.dataSet:
            print(dataLine)
