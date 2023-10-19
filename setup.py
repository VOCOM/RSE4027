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

import matplotlib.pyplot as plt
import numpy as np

def Operations():
    global func
    operationList = [
        "1) Clear Screen",
        "2) Print data table",
        "3) Data Distributions",
        "4) Filtered table",
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
            dataPath = "Milestone1"
        elif project == "2":
            dataPath = "Milestone2"
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

data = Reader.loadData(dataPath + "/train")
func = 0
while Operations():
    os.system("clear")
    if func == "1":
        pass
    if func == "2":
        data.PrintData()
    if func == "3":
        eda.Info(data)
    if func == "4":
        print(data.header)
        func = input("Choose a category to search for: ")
        if func in data.dict.keys():
            distributionTable = eda.Find(data, func)
            eda.PrintDistribution(distributionTable)
            
            xAxis = np.array(list(distributionTable.keys()))
            yAxis = np.array(list(distributionTable.values()))
            print(xAxis)
            print(yAxis)
            plt.bar(xAxis, yAxis)
            plt.show()
        else:
            os.system("clear")
            print("Category not found")

