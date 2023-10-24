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
from FileReader import Reader
from EDA import eda

import pandas
from sklearn import linear_model

import matplotlib.pyplot as plt
import numpy as np

def Operations():
    global func
    operationList = [
        "1) Clear Screen",
        "2) Print data table",
        "3) Data Distributions",
        "4) Filtered table",
        "5) Multiple Regression",
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
os.system("clear")
Menu()

# Step 1: Exploratory Data Analysis
headerFormat = {
    "Passenger ID" : int,
    "Passenger Fare" : float,
    "Ticket Class" : int
}

# data = Reader.loadData(dataPath + "/train")
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
        # Multiple Regression
        regr = linear_model.LinearRegression()
        X = data[["Age", "NumParentChild", "NumSiblingSpouse"]]
        y = data["Survived"]

        regr.fit(X,y)
        # 801,$18,2,234360,0,S,"Milling, Mr. Jacob Christian",48,male,0,0,No
        # 814,$61.9292,1,PC 17485,A20,C,"Duff Gordon, Sir. Cosmo Edmund (""Mr Morgan"")",49,male,1,0,Yes
        predictedSurvival = regr.predict([[49, 1, 0]])
        print(predictedSurvival)
        if predictedSurvival < 0.5:
            print("Passenger will not survive")
        else:
            print("Passenger will survive")
        pass

