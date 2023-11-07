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

import math
import re

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

ignoreMissingDollarState = -1 # -1 is unknown, 0 is don't ignore, 1 is ignore

def str2NaN(value):
    if value == "0":
        value = np.nan
    return value

data["Cabin"] = data["Cabin"].apply(str2NaN)
data["Embarkation Country"] = data["Embarkation Country"].apply(str2NaN)

for x in data.index:
    # Clean Passenger Fare
    try:
        float(data.loc[x, "Passenger Fare"].replace("$",""))
    except ValueError:
        print("Invalid input for Passenger ID "+str(x+1)+", removing value")
        data.loc[x, "Passenger Fare"] = np.nan

    if data.loc[x, "Passenger Fare"][0] != '$':
        if ignoreMissingDollarState == -1:
            while not pandas.isna(data.loc[x, "Passenger Fare"]):
                response = input("Missing $ sign for Passenger ID "+str(x+1)+", keep value(s) for all future occurrence? Y/N: ")
                if 'N' in response.upper():
                    # print("Removing value for Passenger ID "+str(x+1))
                    # data.loc[x, "Passenger Fare"] = np.nan
                    ignoreMissingDollarState = 0
                    break
                if 'Y' in response.upper():
                    ignoreMissingDollarState = 1
                    break
        if ignoreMissingDollarState == 0:
            print("Removing value for Passenger ID "+str(x+1))
            data.loc[x, "Passenger Fare"] = np.nan
    elif data.loc[x, "Passenger Fare"][0] == '$':
        data.loc[x, "Passenger Fare"] = data.loc[x, "Passenger Fare"].replace("$","")
    if not pandas.isna(data.loc[x, "Passenger Fare"]):
        data.loc[x, "Passenger Fare"] = math.floor(float(data.loc[x, "Passenger Fare"]) * 100)/100
        data.loc[x, "Passenger Fare"] = "{:.2f}".format(data.loc[x, "Passenger Fare"]) # Process takes awhile

    # Clean Ticket Class
    try:
        int(data.loc[x, "Ticket Class"])
    except ValueError:
        data.loc[x, "Ticket Class"] = np.nan
    if not pandas.isna(data.loc[x, "Ticket Class"]):
        if int(data.loc[x, "Ticket Class"]) < 1 or int(data.loc[x, "Ticket Class"]) > 3:
            data.loc[x, "Ticket Class"] = np.nan

    # Clean Age
    try:
        data.loc[x, "Age"] = math.floor(float(data.loc[x, "Age"]))
    except ValueError:
        data.loc[x, "Age"] = np.nan
    if not pandas.isna(data.loc[x, "Age"]):
        data.loc[x, "Age"] = "{:.0f}".format(data.loc[x, "Age"])

    # No clean Ticket Number

    # No clean Cabin

    # Clean Embarkation Country
    if not pandas.isna(data.loc[x, "Embarkation Country"]):
        data.loc[x, "Embarkation Country"] = data.loc[x, "Embarkation Country"].upper()
        if len(data.loc[x, "Embarkation Country"]) > 1 or not data.loc[x, "Embarkation Country"].isalpha():
            data.loc[x, "Embarkation Country"] = np.nan

# Without using apply() end

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

