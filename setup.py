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
# - 07/11/23:
#   Clean data added.
#   One-hot encoding added.
# - 10/11/23:
#   EDA Visualization testing added.
# - 12/11/23:
#   Added Survival distributions in the distribution plots
##

import os

import pandas
import numpy as np
from sklearn import linear_model
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from EDA import eda

clearCMD = "cls"

def Operations():
    global func
    global testedSurvivors
    global testedNonSurvivors
    operationList = [
        "1) Clear Screen",
        "2) Print original data table",
        "3) Print extracted data table",
        "4) Print test table",
        "5) Data Distributions",
        "6) Filtered table",
        "7) Logistic Regression",
        "8) K-Nearest Neighbour",
        "9) Prediction Results",
        "E) Exit Program"
    ]

    for operation in operationList:
        print(operation)
    func = input("Operation:")

    if func.strip().capitalize() == "E":
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

def Plots(data):
    x = 0
    maxVal = 0
    label = []
    plotList = [
        '1) Survivor distribution',
        '2) Fare distribution',
        '3) Ticket distribution',
        '4) Country distribution',
        '5) Age distribution',
        '6) Gender distribution',
        '7) Vertical Dependents distribution',
        '8) Horizontal Dependents distribution',
        'E) Return to menu'
    ]

    for plotOption in plotList:
        print(plotOption)
    plotType = input("Plot Type:")
    fig, ax = plt.subplots()
    if plotType == "1":
        label = ['Yes', 'No']
        survived = data["Survived"].value_counts()
        ax.bar(label, survived)
        plt.xlabel('Survived')
    elif plotType == "2":
        for val in data['Passenger Fare']:
            if val > maxVal:
                maxVal = val
        maxVal = round(maxVal / 100)
        data['Passenger Fare'].plot(kind="hist", edgecolor='white', bins=maxVal)
        plt.xlabel('Fare Amount')
    elif plotType == "3":
        label = list(data["Ticket Class"].unique())
        ticketSurvival = {
            'Survived' : [0, 0, 0],
            'Not Survived' : [0, 0, 0]
        }
        while x < len(data):
            if data.loc[x,'Survived']:
                ticketSurvival['Survived'][data.loc[x,'Ticket Class']-1] += 1
            else:
                ticketSurvival['Not Survived'][data.loc[x,'Ticket Class']-1] += 1
            x+=1
        bottom = np.zeros(3)
        for boolean, count in ticketSurvival.items():
            ax.bar(label, count, label=boolean, bottom=bottom)
            bottom+=count
        plt.xlabel('Ticket Class')
    elif plotType == "4":
        label = ['Q', 'C', 'S']
        countrySurvival = {
            'Survived' : [0, 0, 0],
            'Not Survived' : [0, 0, 0]
        }
        while x < len(data):
            if data.loc[x, 'Survived']:
                if data.loc[x,'Q']:
                    countrySurvival['Survived'][0]+=1
                if data.loc[x,'C']:
                    countrySurvival['Survived'][1]+=1
                if data.loc[x,'S']:
                    countrySurvival['Survived'][2]+=1
            else:
                if data.loc[x,'Q']:
                    countrySurvival['Not Survived'][0]+=1
                if data.loc[x,'C']:
                    countrySurvival['Not Survived'][1]+=1
                if data.loc[x,'S']:
                    countrySurvival['Not Survived'][2]+=1
            x+=1
        bottom = np.zeros(3)
        for boolean, count in countrySurvival.items():
            ax.bar(label, count, label=boolean, bottom=bottom)
            bottom+=count
        plt.xlabel('Embarkation Country')
    elif plotType == "5":
        age = data['Age'].value_counts()
        label = list(age.keys())
        ax.bar(label,age)
        plt.xlabel('Passenger Age')
    elif plotType == "6":
        label = ['Male', 'Female']
        genderSurvival = {
            'Survival' : [0, 0],
            'Not Survival' : [0, 0]
        }
        while x < len(data):
            if data.loc[x, 'Survived']:
                genderSurvival['Survival'][data.loc[x,'Gender']]+=1
            else:
                genderSurvival['Not Survival'][data.loc[x,'Gender']]+=1
            x+=1
        bottom = np.zeros(2)
        for boolean, count in genderSurvival.items():
            ax.bar(label, count, label=boolean, bottom=bottom)
            bottom+=count
        plt.xlabel('Gender')
    elif plotType == "7":
        for val in data['NumParentChild']:
            if val > maxVal:
                maxVal = val
        label = list(data["NumParentChild"].unique())
        VertDependantSurvival = {
            'Survived' : np.zeros(maxVal+1, dtype=int),
            'Not Survived' : np.zeros(maxVal+1, dtype=int)
        }
        while x < len(data):
            if data.loc[x,'Survived']:
                VertDependantSurvival['Survived'][data.loc[x,'NumParentChild']]+=1
            else:
                VertDependantSurvival['Not Survived'][data.loc[x,'NumParentChild']]+=1
            x+=1
        bottom = np.zeros(maxVal+1)
        for boolean, count in VertDependantSurvival.items():
            ax.bar(label, count, label=boolean, bottom=bottom)
            bottom+=count
        plt.xlabel('Number of Parents & Children')
    elif plotType == "8":
        for val in data['NumSiblingSpouse']:
            if val > maxVal:
                maxVal = val
        label = list(data["NumSiblingSpouse"].unique())
        VertDependantSurvival = {
            'Survived' : np.zeros(maxVal+1, dtype=int),
            'Not Survived' : np.zeros(maxVal+1, dtype=int)
        }
        while x < len(data):
            if data.loc[x,'Survived']:
                VertDependantSurvival['Survived'][data.loc[x,'NumSiblingSpouse']]+=1
            else:
                VertDependantSurvival['Not Survived'][data.loc[x,'NumSiblingSpouse']]+=1
            x+=1
        bottom = np.zeros(maxVal+1)
        for boolean, count in VertDependantSurvival.items():
            ax.bar(label, count, label=boolean, bottom=bottom)
            bottom+=count
        plt.xlabel('Number of Siblings & Spouses')
    else:
        os.system(clearCMD)
        return False
    ax.legend(loc="upper right")
    plt.ylabel('Number of Passengers')
    plt.show()
    plt.close()
    os.system(clearCMD)
    return True

def PredictionPlots(data):
    x = 0
    maxVal = 0
    label = []
    plotList = [
        '1) Fare distribution',
        '2) Ticket distribution',
        '3) Country distribution',
        '4) Age distribution',
        '5) Gender distribution',
        '6) Vertical Dependents distribution',
        '7) Horizontal Dependents distribution',
        'E) Return to menu'
    ]
    for plotOption in plotList:
        print(plotOption)
    plotType = input("Plot Type:")
    fig, ax = plt.subplots()
    if plotType == "1":
        pass
    if plotType == "2":
        label = list(data["Ticket Class"].unique())
        count = list(data["Ticket Class"].value_counts())
        ax.bar(label, count)
        plt.xlabel("Ticket Class")
    if plotType == "3":
        pass
    if plotType == "4":
        pass
    if plotType == "5":
        pass
    if plotType == "6":
        pass
    if plotType == "7":
        pass
    if plotType == "8":
        pass
    if plotType == "E":
        pass
    ax.legend(loc="upper right")
    plt.ylabel('Number of Passengers')
    plt.show()
    plt.close()
    os.system(clearCMD)

def LogisticRegression():
    # Logistic Regression
    global testedSurvivors
    global testedNonSurvivors
    regr = linear_model.LogisticRegression(max_iter=1000)
    inputParameters = [
        'Passenger Fare',
        'Ticket Class',
        'Age',
        'Gender',
        'NumParentChild',
        'NumSiblingSpouse',
        'Q',
        'C',
        'S'
    ]
    X = extractedData[inputParameters].values
    y = list(extractedData['Survived'])
    regr = regr.fit(X,y)

    predictions = []

    i = 0
    while i < len(test):
        inFare = test['Passenger Fare'][i]
        inClass = test['Ticket Class'][i]
        inAge = test['Age'][i]
        inGender = test['Gender'][i]
        inParent = test['NumParentChild'][i]
        inSibling = test['NumSiblingSpouse'][i]
        inQ = test['Q'][i]
        inC = test['C'][i]
        inS = test['S'][i]
        predictedSurvival = regr.predict([[inFare, inClass, inAge, inGender, inParent, inSibling, inQ, inC, inS]])
        predictions.append(predictedSurvival)
        if predictedSurvival:
            testedSurvivors.loc[len(testedSurvivors.index)] = test.loc[i]
        else:
            testedNonSurvivors.loc[len(testedNonSurvivors.index)] = test.loc[i]
        i += 1

    print("Logistic Regression Metrics")
    precision = precision_score(list(test['Survived']), predictions)
    recall = recall_score(list(test['Survived']), predictions)
    fScore = f1_score(list(test['Survived']), predictions)
    print("Precision: {:.5f}".format(precision))
    print("Recall:    {:.5f}".format(recall))
    print("F1 Score:  {:.5f}".format(fScore))
    print()

def FilteredTable():
    header = list(extractedData.keys())
    print(header)
    func = input("Choose a category to search for: ")
    if func in header:
        print(extractedData[func].to_string())
    else:
        os.system(clearCMD)
        print("Category not found\n")

def KNearestNeigbour():
    pass

def PrintPredictionResults():
    global testedSurvivors
    global testedNonSurvivors
    func = 0
    resultsOptions = [
        "1) Survivors",
        "2) Non Survivors",
        "E) Exit"
    ]
    os.system(clearCMD)
    if len(testedSurvivors) == 0 or len(testedNonSurvivors) == 0:
        print("No data detected! Apply a model first.")
        return
    while func != "E":
        for options in resultsOptions:
            print(options)
        func = input("Predictions to list:").capitalize()
        os.system(clearCMD)
        if func == "1":
            print("Survivors\n", testedSurvivors.to_string(), '\n')
            PredictionPlots(testedSurvivors)
        if func == "2":
            print("Non Survivors\n", testedNonSurvivors.to_string(), '\n')
            PredictionPlots(testedNonSurvivors)

dataPath = ""
Menu()

rawData = pandas.read_csv(dataPath + "/train/MS_1_Scenario_train.csv")
test = pandas.read_csv(dataPath + "/test/MS_1_Scenario_test.csv")

rawData = eda.Clean(rawData)
test = eda.Clean(test)

extractedData = eda.Extract(rawData)
test = eda.Extract(test)

testedSurvivors = pandas.DataFrame(columns=test.columns)
testedNonSurvivors = pandas.DataFrame(columns=test.columns)

func = 0
while Operations():
    os.system(clearCMD)
    if func == "1":
        pass
    if func == "2":
        print(rawData.to_string(), "\n")
    if func == "3":
        print(extractedData.to_string(), "\n")
    if func == "4":
        print(test.to_string(), "\n")
    if func == "5":
        os.system(clearCMD)
        print("1) Original data")
        print("2) Extracted data")
        print("3) Test data")
        func = input("Data to be analysed:")
        if func == "1":
            data = rawData
        elif func == "2":
            data = extractedData
        else:
            data = test
        print()
        while Plots(data):
            pass
    if func == "6":
        FilteredTable()
    if func == "7":
        LogisticRegression()
    if func == "8":
        KNearestNeigbour()
    if func == "9":
        PrintPredictionResults()