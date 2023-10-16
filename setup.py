import os
from FileReader import Reader
from EDA import eda


def Options():
    index = 0
    
    global func 
    print(index,") Print data table")
    index += 1
    print(index,") Find All based on header category")
    func = input("Functions:")

    if func == "E" or func == "e":
        return False
    else:
        return True

print("1) Milestone 1")
print("2) Milestone 2")
project = input("Select a project: ")

if project == "1":
    dataPath = "Milestone1"
elif project == "2":
    dataPath = "Milestone2"

# Step 1: Exploratory Data Analysis
headerFormat = {
    "Passenger ID" : int,
    "Passenger Fare" : float,
    "Ticket Class" : int
}

# test = Reader.loadCSV(dataPath + "/train/MS_1_Scenario_train.csv", True)
# test.PrintData()
# exit()

data = Reader.loadData(dataPath + "/train")
func = 0
while Options():
    index = 0
    if func == "0":
        data.PrintData()
    if func == "1":
        print(data.header)
        func = input("Choose a category to search for: ")
        while index < len(data.header):
            if func == data.header[index]:
                category = data.header[index]
                break
            index += 1
        inputText = input("Survived?: ")
        eda.FindAll(data, index, inputText)
# data.PrintData()
# eda.FindAll(data, 11, "Yes")