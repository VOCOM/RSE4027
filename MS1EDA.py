import os
from setup import Setup, EDAOperations, Plots, VisualizeEda
from EDA.eda import Clean, Extract

clearCMD = 'cls'
rawTrainData, rawTestData = Setup()

Clean(rawTrainData)
Clean(rawTestData)

extractedTrainData = Extract(rawTrainData)
extractedTestData = Extract(rawTestData)

userInput = EDAOperations()

while userInput != "E":
    os.system(clearCMD)
    if userInput == "2":
        print("Input Training Data")
        print(rawTrainData.to_string(), "\n")
    if userInput == "3":
        print("Extracted Training Data")
        print(extractedTrainData.to_string(), "\n")
    if userInput == "4":
        print("Input Testing Data")
        print(rawTestData.to_string(), "\n")
    if userInput == "5":
        print("Extracted Testing Data")
        print(extractedTestData.to_string(), "\n")
    if userInput == "6":
        os.system(clearCMD)
        print("1) Extracted data")
        print("2) Test data")
        userInput = input("Data to be analysed:")
        if userInput == "1":
            data = extractedTrainData
        else:
            data = extractedTestData
        print()
        while Plots(data):
            pass
    if userInput == "7":
        visualizeList = [
            "1) Ticket Class vs Survived",
            "2) Embarkation Country vs Survived",
            "3) Gender Class vs Survived",
            "4) NumSiblingSpouse vs Survived",
            "5) NumParentChild vs Survived", 
            "6) Correlation of all numerical data columns",
            "7) Percentage of NaN in data column"
            
        ]
        for visualize in visualizeList:
            print(visualize)
        edaVisualize = input("Visualize plot: ")
        VisualizeEda(rawTrainData, edaVisualize)
    
    userInput = EDAOperations()