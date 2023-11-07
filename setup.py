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
        "3) Print test table",
        "4) Data Distributions",
        "5) Filtered table",
        "6) Manual Logistic Regression",
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

def Plots():
    maxVal = 0
    plotList = [
        '1) Survivor distribution',
        '2) Fare distribution',
        '3) Ticket distribution',
        '4) Country distribution',
        '5) Age distribution',
        '6) Gender distribution',
        '7) Vertical Dependents distribution',
        '8) Horizontal Dependents distribution'
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
        label = ['1', '2', '3']
        ticketClass = data["Ticket Class"].value_counts()
        ax.bar(label, ticketClass)
        plt.xlabel('Ticket Class')
    elif plotType == "4":
        embarkation = data["Embarkation Country"].value_counts()
        label = ['S','C','Q','0']
        ax.bar(label,embarkation)
        plt.xlabel('Embarkation Country')
    elif plotType == "5":
        # TODO
        age = data['Age'].value_counts()
        label = list(data['Age'].keys())
        print(label)
        input()
        # data['Age'].plot(kind="hist", edgecolor='white', bins=10)
        ax.bar(label,age)
        plt.xlabel('Age')
    elif plotType == "6":
        label = ['Male', 'Female']
        gender = data['Gender'].value_counts()
        ax.bar(label,gender)
        plt.xlabel('Gender')
    elif plotType == "7":
        for val in data['NumParentChild']:
            if val > maxVal:
                maxVal = val
        data['NumParentChild'].plot(kind="hist", edgecolor='white', bins=maxVal)
        plt.xlabel('Number of Parents & Children')
    elif plotType == "8":
        for val in data['NumSiblingSpouse']:
            if val > maxVal:
                maxVal = val
        data['NumSiblingSpouse'].plot(kind="hist", edgecolor='white', bins=8)
        plt.xlabel('Number of Siblings & Spouses')
    else:
        return
    plt.ylabel('Number of Passengers')
    plt.show()
    os.system(clearCMD)

dataPath = ""
Menu()

data = pandas.read_csv(dataPath + "/train/MS_1_Scenario_train.csv")
test = pandas.read_csv(dataPath + "/test/MS_1_Scenario_test.csv")

data = eda.Clean(data)
test = eda.Clean(test)

func = 0
while Operations():
    os.system(clearCMD)
    if func == "1":
        pass
    if func == "2":
        print(data.to_string(), "\n")
    if func == "3":
        print(test.to_string(), "\n")
    if func == "4":
        Plots()
    if func == "5":
        header = list(data.keys())
        print(header)
        func = input("Choose a category to search for: ")
        if func in header:
            print(data[func].to_string())
        else:
            os.system(clearCMD)
            print("Category not found\n")
    if func == "6":
        # Logistic Regression
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

        
        # print("Enter Passenger details\n")
        # inFare = round(float(input("Ticket Price (USD):")), 2)
        # inClass = int(input("Ticket Class (1,2,3):"))
        # inEmbark = input("Embarkation Country (C,S,Q):")
        # if 'C' in inEmbark.capitalize()[0] or 'S' in inEmbark.capitalize()[0]:
        #     inEmbark = ord(inEmbark.capitalize()[0])
        # else:
        #     inEmbark = ord('Q')
        # inAge = int(input("Age:"))
        # inGender = input("Gender [M/F]:")
        # if inGender.capitalize()[0] == 'F':
        #     inGender = 1
        # else:
        #     inGender = 0
        # inParent = int(input("Number of Parents and Siblings:"))
        # inSibling = int(input("Number of Siblings and Spouses:"))
        # print("\nWeights:\n")
        # print("Passenger Fare:", inFare, "Ticket Class:", inClass, "Embarkation Country:", chr(inEmbark),
        #       "Age:", inAge, "Gender:", inGender, "Number of Parents & Siblings:", inParent, "Number of Siblings and Spouses:", inSibling)
        inFare = test['Passenger Fare']
        inClass = test['Ticket Class']
        inEmbark = test['Embarkation Country']
        inAge = test['Age']
        inGender = test['Gender']
        inParent = test['NumParentChild']
        inSibling = test['NumSiblingSpouse']
        print("Passenger Fare:", inFare, "Ticket Class:", inClass, "Embarkation Country:", chr(inEmbark),
              "Age:", inAge, "Gender:", inGender, "Number of Parents & Siblings:", inParent, "Number of Siblings and Spouses:", inSibling)
        predictedSurvival = regr.predict([[inFare, inClass, inEmbark, inAge, inGender, inParent, inSibling]])
        if predictedSurvival:
            print("Passenger will survive")
        else:
            print("Passenger will not survive")
        print()
        pass

