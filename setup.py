import os
from EDA import eda

print("1) Milestone 1")
print("2) Milestone 2")
project = input("Select a project: ")

if project == "1":
    dataPath = "Milestone1/data"
elif project == "2":
    dataPath = "Milestone2/data"

eda.loadData(dataPath)
# eda.extractData()