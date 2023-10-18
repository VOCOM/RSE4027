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
            processedline = processedline.split(",")
            if len(processedline) != self.headerSize + self.namedFlag:
                print("There is missing information on line ", str(index))
            
            if self.namedFlag:
                processedline = self.CleanName(processedline)
            
            self.dataSet.append(processedline)
            # print(processedline)
            index += 1

    def CleanName(self, line):
        processedString = line
        processedString[6] = processedString[6].strip().replace("\"", "")
        processedString[7] = processedString[7].strip().replace("\"", "")
        name = processedString[6] + ", " + processedString[7]
        processedString[6] = name
        index = 8
        while index < len(processedString):
            processedString[index - 1] = processedString[index]
            index += 1
        processedString.pop()
        return processedString


    def PrintData(self):
        print(self.header)
        for dataLine in self.dataSet:
            print(dataLine)
