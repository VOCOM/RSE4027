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
    os.system("clear")
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
            os.system("clear")

dataPath = ""
Menu()

data = pandas.read_csv(dataPath + "/train/MS_1_Scenario_train.csv")

data["Gender"] = data["Gender"].replace({"female" : 1}, regex=True)
data["Gender"] = data["Gender"].replace({"male" : 0}, regex=True)
data["Survived"] = data["Survived"].replace({"Yes" : 1}, regex=True)
data["Survived"] = data["Survived"].replace({"No" : 0}, regex=True)
data["NumParentChild"].convert_dtypes(convert_integer=True)
data["NumSiblingSpouse"].convert_dtypes(convert_integer=True)
data["Age"].convert_dtypes(convert_integer=True)

func = 0
while Operations():
    os.system("clear")
    if func == "1":
        pass
    if func == "2":
        print(data.to_string(), "\n")
    if func == "3":
        data["Survived"].plot(kind="hist")
    if func == "4":
        header = list(data.keys())
        print(header)
        func = input("Choose a category to search for: ")
        if func in header:
            print(data[func].to_string())
        else:
            os.system("clear")
            print("Category not found\n")
    if func == "5":
        # Logistic Regression
        # regr = linear_model.LinearRegression()
        regr = linear_model.LogisticRegression()
        X = data[['Age', 'NumParentChild', 'NumSiblingSpouse']].values
        y = data['Survived']
        regr.fit(X,y)

        # 801,$18,2,234360,0,S,"Milling, Mr. Jacob Christian",48,male,0,0,No
        # 814,$61.9292,1,PC 17485,A20,C,"Duff Gordon, Sir. Cosmo Edmund (""Mr Morgan"")",49,male,1,0,Yes
        print("Enter Passenger details")
        inAge = int(input("Age: "))
        inParent = int(input("Number of Parents and Siblings: "))
        inSibling = int(input("Number of Siblings and Spouses: "))
        predictedSurvival = regr.predict([[inAge, inParent, inSibling]])
        if predictedSurvival:
            print("Passenger will survive")
        else:
            print("Passenger will not survive")
        print()
        pass

