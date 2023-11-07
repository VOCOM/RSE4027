from FileReader.DataTypes import DATA

import pandas
import numpy as np
import math
import re

def Find(data, category):
    distributionTable = {}
    for entry in data.dict.get(category):
        if entry in distributionTable.keys():
            distributionTable[entry] += 1
        else:
            distributionTable[entry] = 1
    return distributionTable

def Info(data):
    for header in data.dict.keys():
        print(header, len(data.dict.get(header)), "entries")
    print()
        
    for header in data.dict.keys():
        print(header)
        distributionTable = Find(data, header)
        PrintDistribution(distributionTable)

def PrintDistribution(distributionTable):
        for distribution in distributionTable.items():
            print(distribution)
        print()

def str2NaN(value):
    if value == "0":
        value = np.nan
    return value

def Clean(data):
    data["Gender"] = data["Gender"].replace({"female" : 1}, regex=True)
    data["Gender"] = data["Gender"].replace({"male" : 0}, regex=True)
    data["Survived"] = data["Survived"].replace({"Yes" : 1}, regex=True)
    data["Survived"] = data["Survived"].replace({"No" : 0}, regex=True)
    data["NumParentChild"] = data["NumParentChild"].astype(int)
    data["NumSiblingSpouse"] = data["NumSiblingSpouse"].astype(int)
    data["Age"] = data["Age"].astype(float)
    data.insert(len(data.columns), "Abnormal", False)
    data["Cabin"] = data["Cabin"].apply(str2NaN)
    data["Embarkation Country"] = data["Embarkation Country"].apply(str2NaN)

    ignoreMissingDollarState = -1 # -1 is unknown, 0 is don't ignore, 1 is ignore

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
        else:
            data.loc[x, 'Abnormal'] = True

        # Clean Ticket Class
        try:
            int(data.loc[x, "Ticket Class"])
        except ValueError:
            data.loc[x, "Ticket Class"] = np.nan
        if not pandas.isna(data.loc[x, "Ticket Class"]):
            if int(data.loc[x, "Ticket Class"]) < 1 or int(data.loc[x, "Ticket Class"]) > 3:
                data.loc[x, "Ticket Class"] = np.nan
        else:
            data.loc[x, 'Abnormal'] = True

        # Clean Age
        try:
            data.loc[x, "Age"] = math.floor(float(data.loc[x, "Age"]))
        except ValueError:
            data.loc[x, "Age"] = np.nan
        if data.loc[x, "Age"] < 1:
            data.loc[x, 'Abnormal'] = True

        # No clean Ticket Number

        # No clean Cabin

        # Clean Embarkation Country
        if not pandas.isna(data.loc[x, "Embarkation Country"]):
            data.loc[x, "Embarkation Country"] = data.loc[x, "Embarkation Country"].upper()
            if len(data.loc[x, "Embarkation Country"]) > 1 or not data.loc[x, "Embarkation Country"].isalpha():
                data.loc[x, "Embarkation Country"] = np.nan
        else:
            data.loc[x, "Abnormal"] = True

    data["Passenger Fare"] = data["Passenger Fare"].astype(float)
    return data

def Extract(data):
    normalData = pandas.DataFrame(columns=data.columns)
    i = 0
    while i < len(data):
        if data.loc[i, 'Abnormal'] == False:
            normalData.loc[len(normalData.index)] = data.loc[i]
        i += 1
    normalData.drop('Abnormal', axis='columns', inplace=True)
    return normalData

