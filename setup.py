## 
# @author Muhammad Syamim Bin Shamsulbani
# @brief Machine learning & Artificial Intelligence Program for RSE4207
# @version 0.5
# @date 16/10/23
#
# Changelog:
# - 16/10/23:
#   Find function added.
#   User Interface added.
##

import os

import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

from EDA import eda

clearCMD = "cls"

def Operations():
    global func
    operationList = [
        "1) Clear Screen",
        "2) Print data table",
        "3) Data Distributions",
        "4) Filtered table",
        "5) Manual Logistic Regression",
        "E) Exit Program"
    ]

    for operation in operationList:
        print(operation)
    func = input("Operation:")

    if func.capitalize() == "E":
        return False
    else:
        return True

def Menu():
    global dataPath
    os.system(clearCMD)
    projectList = [
        "1) Milestone 1",
        "2) Milestone 2",
        "E) Exit Program"
    ]

    loop = True

    while loop:
        loop = False
        for project in projectList:
            print(project)
        project = input("Select a project: ")

        if project == "1":
            dataPath = "./Milestone1"
        elif project == "2":
            dataPath = "./Milestone2"
        elif project.capitalize() == "E":
            exit()
        else:
            loop = True
        os.system(clearCMD)

dataPath = ""
Menu()

data = pandas.read_csv(dataPath + "/train/MS_1_Scenario_train.csv")

data = eda.Clean(data)

func = 0
while Operations():
    os.system(clearCMD)
    if func == "1":
        pass
    if func == "2":
        print(data.to_string(), "\n")
    if func == "3":
        data["Survived"].groupby(level=0).hist(bins=2, grid=False)
        plt.show()
    if func == "4":
        header = list(data.keys())
        print(header)
        func = input("Choose a category to search for: ")
        if func in header:
            print(data[func].to_string())
        else:
            os.system(clearCMD)
            print("Category not found\n")
    if func == "5":
        # Logistic Regression
        # regr = linear_model.LinearRegression()
        regr = linear_model.LogisticRegression(max_iter=1000)
        inputParameters = [
            'Passenger Fare',
            'Ticket Class',
            'Embarkation Country',
            'Age',
            'Gender',
            'NumParentChild',
            'NumSiblingSpouse'
        ]
        X = data[inputParameters].values
        y = data['Survived']
        regr.fit(X,y)

        # 801,$18,2,234360,0,S,"Milling, Mr. Jacob Christian",48,male,0,0,No [Predicted No]
        # 814,$61.9292,1,PC 17485,A20,C,"Duff Gordon, Sir. Cosmo Edmund (""Mr Morgan"")",49,male,1,0,Yes [Predicted No]
        # 876,$44,2,230136,F4,S,"Becker, Miss. Marion Louise",4,female,2,1,Yes [Predicted Yes]
        print("Enter Passenger details\n")
        inFare = round(float(input("Ticket Price (USD):")), 2)
        inClass = int(input("Ticket Class (1,2,3):"))
        inEmbark = input("Embarkation Country (C,S,Q):")
        if 'C' in inEmbark.capitalize()[0] or 'S' in inEmbark.capitalize()[0]:
            inEmbark = ord(inEmbark.capitalize()[0])
        else:
            inEmbark = ord('Q')
        inAge = int(input("Age:"))
        inGender = input("Gender [M/F]:")
        if inGender.capitalize()[0] == 'F':
            inGender = 1
        else:
            inGender = 0
        inParent = int(input("Number of Parents and Siblings:"))
        inSibling = int(input("Number of Siblings and Spouses:"))
        print("\nWeights:\n")
        print("Passenger Fare:", inFare, "Ticket Class:", inClass, "Embarkation Country:", chr(inEmbark),
              "Age:", inAge, "Gender:", inGender, "Number of Parents & Siblings:", inParent, "Number of Siblings and Spouses:", inSibling)
        predictedSurvival = regr.predict([[inFare, inClass, inEmbark, inAge, inGender, inParent, inSibling]])
        if predictedSurvival:
            print("Passenger will survive")
        else:
            print("Passenger will not survive")
        print()
        pass

