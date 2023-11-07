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
        "6) Logistic Regression",
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
        age = data['Age'].value_counts()
        label = list(age.keys())
        ax.bar(label,age)
        plt.xlabel('Passenger Age')
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
        os.system(clearCMD)
        return False
    plt.ylabel('Number of Passengers')
    plt.show()
    os.system(clearCMD)
    return True

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
        while Plots():
            pass
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
        regr = regr.fit(X,y)

        confusionMatrix = pandas.DataFrame({
            'Positive' : [0, 0],
            'Negative' : [0, 0]
        }, index = ['True', 'False'])

        i = 0
        while i < len(test):
            inFare = test['Passenger Fare'][i]
            inClass = test['Ticket Class'][i]
            inEmbark = test['Embarkation Country'][i]
            inAge = test['Age'][i]
            inGender = test['Gender'][i]
            inParent = test['NumParentChild'][i]
            inSibling = test['NumSiblingSpouse'][i]
            # print("Passenger Fare:", inFare, "Ticket Class:", inClass, "Embarkation Country:", chr(inEmbark),
            #     "Age:", inAge, "Gender:", inGender, "Number of Parents & Siblings:", inParent, "Number of Siblings and Spouses:", inSibling)
            predictedSurvival = regr.predict([[inFare, inClass, inEmbark, inAge, inGender, inParent, inSibling]])
            
            if predictedSurvival:
                if test['Survived'][i]: #TP
                    confusionMatrix.at['True','Positive'] += 1
                else: #FP
                    confusionMatrix.at['False','Positive'] += 1
            else:
                if test['Survived'][i]: #FN
                    confusionMatrix.at['False','Negative'] += 1
                else: #TN
                    confusionMatrix.at['True','Negative'] += 1
            i += 1
        print("\n", confusionMatrix, "\n")
        precision = confusionMatrix.at['True','Positive'] / (confusionMatrix.at['True','Positive'] + confusionMatrix.at['True','Negative'])
        recall = confusionMatrix.at['True','Positive'] / (confusionMatrix.at['True','Positive'] + confusionMatrix.at['False','Negative'])
        print("Precision:", precision)
        print("Recall:", recall, "\n")
