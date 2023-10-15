import os
from FileReader import Reader
from EDA import eda

print("1) Milestone 1")
print("2) Milestone 2")
project = input("Select a project: ")

if project == "1":
    dataPath = "Milestone1"
elif project == "2":
    dataPath = "Milestone2"

# Step 1: Exploratory Data Analysis
data = Reader.loadData(dataPath + "/train")

# data.PrintData()
eda.FindAll(data, 11, "Yes")