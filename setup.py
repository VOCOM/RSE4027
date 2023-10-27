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

# Using apply() start
# Pros: Apply custom functions to every elements in the selected column (eg, Passenger Fare, Ticket Class, etc.)
# Cons: Method of tracking passenger ID
#       Custom functions does not accept >1 parameters

# passengerIndex = 0 # Primitive method of tracking Passenger ID, to be changed
# def roundFare(fare):
#     global passengerIndex 
#     passengerIndex = passengerIndex + 1
#     fare = re.sub("[^0-9.$€£¥₣₹₽₾₺₼₸₴₷฿원₫₮₯₱₳₵₲₪₰]","",fare)
#     for i in "€£¥₣₹₽₾₺₼₸₴₷฿원₫₮₯₱₳₵₲₪₰":
#         if i in fare:
#             fare = input("Other currency symbol detected, key in replacement value (in $) for Passenger ID " + str(passengerIndex) + ": ")
#     if fare[0] == '$':
#         fare = fare.replace('$','')
#     # if fare[0] != '$':
#     #     response = input("$ header missing for Passenger ID " + str(passengerIndex) + ", key in replacement value? Y/N: ")
#     #     if response.upper == 'N':
#     #         fare = input("Please key in replacement value for Passenger ID " + str(passengerIndex) + " (in $): ")
#     #     if response.upper == 'Y':
#     #         fare = fare.replace('$','')
#     #     else:
#     #         print("Invalid input, keeping values for Passenger ID " + str(passengerIndex))
#     try:
#         fare = float(fare)
#     except ValueError:
#         fare = -1.00
#         fare = input("Invalid values detected, key in replacement value (in $) for Passenger ID " + str(passengerIndex) + ": ")
#     return float(math.floor(fare * 100)/100)

# def roundAge(age):
#     return int(age)

# data["Passenger Fare"] = data["Passenger Fare"].apply(roundFare)
# # data["Passenger ID"] = data.apply(rowIndex, axis = 1)
# data["Age"] = data["Age"].apply(roundAge)

# Using apply() end

# Without using apply() start
# Pros: Uses loc member to access specific elements within column, if condition can be used on the element meeting the condition
# Cons: Seems to take longer for every iteration implemented

unknownVal = -1

for x in data.index:
    # Clean Passenger Fare
    data.loc[x, "Passenger Fare"] = re.sub("[^0-9.$]","",data.loc[x, "Passenger Fare"])
    if '$' not in str(data.loc[x, "Passenger Fare"]):
        while True:
            response = input("Missing $ sign for Passenger ID "+str(x+1)+", keep the value? Y/N: ")
            if 'N' in response.upper():
                try:
                    data.loc[x, "Passenger Fare"] = float(input("Input Passenger Fare for Passenger ID "+str(x+1)+" (in $): "))
                except ValueError:
                    print("Invalid input for Passenger Fare, setting value to unknownVal")
                    data.loc[x, "Passenger Fare"] = unknownVal
                break
            if 'Y' in response.upper():
                break
    elif '$' in str(data.loc[x, "Passenger Fare"]):
        data.loc[x, "Passenger Fare"] = data.loc[x, "Passenger Fare"].replace("$","")
    data.loc[x, "Passenger Fare"] = math.floor(float(data.loc[x, "Passenger Fare"]) * 100)/100
    data.loc[x, "Passenger Fare"] = "{:.2f}".format(data.loc[x, "Passenger Fare"]) # Process takes awhile

    # Clean Ticket Class
    try:
        int(data.loc[x, "Ticket Class"])
    except ValueError:
        data.loc[x, "Ticket Class"] = unknownVal
    if int(data.loc[x, "Ticket Class"]) < 1 or int(data.loc[x, "Ticket Class"]) > 3:
        data.loc[x, "Ticket Class"] = unknownVal

    # Clean Age
    try:
        data.loc[x, "Age"] = math.floor(float(data.loc[x, "Age"]))
    except ValueError:
        data.loc[x, "Age"] = unknownVal
    data.loc[x, "Age"] = "{:.0f}".format(data.loc[x, "Age"])
    # data.loc[x, "Age"] = re.sub("[^0-9.]","",str(data.loc[x, "Age"]))
    # data.loc[x, "Age"] = int(math.floor(float(data.loc[x, "Age"])))

    # No clean Ticket Number

    # No clean Cabin

    # Clean Embarkation Country
    if data.loc[x, "Embarkation Country"].__len__() > 1:
        data.loc[x, "Embarkation Country"] = input("Invalid Embarkation Country for Passenger ID "+str(x+1)+", key in correct value (single alphabetical character): ")
    # if data.loc[x, "Embarkation Country"].str.


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

