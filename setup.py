## 
# @author Muhammad Syamim Bin Shamsulbani
# @brief Machine learning & Artificial Intelligence Program for RSE4207
# @version 0.5
# @date 16/10/23
#
# Changelog:
# - 16/10/23:
#   Find userInputtion added.
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
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import matplotlib.pyplot
import seaborn as sns

from EDA import eda

clearCMD = "cls"

def Setup():
    config = open('config.txt')
    for line in config.readlines():
        if "Milestone1_train_csv: " in line:
            trainingDataPath = line.strip().split("\"")[1].replace("\\","/")
        if "Milestone1_test_csv: " in line:
            testDataPath = line.strip().split("\"")[1].replace("\\","/")
    trainData = pandas.read_csv(trainingDataPath)
    testData = pandas.read_csv(testDataPath)
    return trainData, testData

def EDAOperations():
    userInput = ''
    operationList = [
        "1) Clear Screen",
        "2) Print original data table",
        "3) Print extracted data table",
        "4) Print original test table",
        "5) Print extracted test table",
        "6) Data Distributions",
        "7) EDA Visualization",
        "E) Exit Program"
    ]

    for operation in operationList:
        print(operation)
    userInput = input("Operation:").capitalize()

    return userInput

def Operations():
    global userInput
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
    userInput = input("Operation:")

    if userInput.strip().capitalize() == "E":
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
        plt.close()
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
    plotType = input("Plot Type:").capitalize()
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
    if plotType == "E" or plotType == "":
        plt.close()
        return plotType
    plt.ylabel('Number of Passengers')
    plt.show()
    plt.close()
    os.system(clearCMD)
    return plotType

def LogisticRegression():
    # Logistic Regression
    global lastAppliedModel
    lastAppliedModel = 'Logistic Regression'
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

    actualVal = list(test['Survived'].values)
    predictions_rounded = np.round(predictions).astype(int)
    mae, mse, rmse = eda.ErrorCalc(predictions_rounded, actualVal)

    print("Logistic Regression Metrics")
    precision = precision_score(list(test['Survived']), predictions)
    recall = recall_score(list(test['Survived']), predictions)
    fScore = f1_score(list(test['Survived']), predictions)
    print("Precision: {:.5f}".format(precision))
    print("Recall:    {:.5f}".format(recall))
    print("F1 Score:  {:.5f}".format(fScore))
    print("MAE:       {:.5f}".format(mae))
    print("MSE:       {:.5f}".format(mse))
    print("RMSE:      {:.5f}".format(rmse))
    print()

def FilteredTable():
    header = list(extractedData.keys())
    print(header)
    userInput = input("Choose a category to search for: ")
    if userInput in header:
        print(extractedData[userInput].to_string())
    else:
        os.system(clearCMD)
        print("Category not found\n")

def KNearestNeigbour():
    # K Nearest Neighbor
    global lastAppliedModel
    global testedSurvivors
    global testedNonSurvivors
    testedSurvivors = testedSurvivors.iloc[0:0]
    testedNonSurvivors = testedNonSurvivors.iloc[0:0]
    lastAppliedModel = 'K-Nearest Neighbour'
    K = 200

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
    
    knn_model = KNeighborsRegressor(n_neighbors = K)
    knn_model.fit(X, y)
    knn_model.feature_names_in_ = inputParameters
    
    actualVal = list(test['Survived'].values)
    predictions = knn_model.predict(test[inputParameters]) 

    predictions_rounded = np.round(predictions).astype(int)

    i = 0
    for prediction in predictions_rounded:
        if prediction:
            testedSurvivors.loc[len(testedSurvivors)] = test.loc[i]
        else:
            testedNonSurvivors.loc[len(testedNonSurvivors)] = test.loc[i]
        i += 1

    print("KNN Metrics")
    precision = precision_score(actualVal, predictions_rounded)
    recall = recall_score(actualVal, predictions_rounded)
    fScore = f1_score(actualVal, predictions_rounded)

    # r2 = r2_score(actualVal, predictions_rounded)
    mae, mse, rmse = eda.ErrorCalc(predictions_rounded, actualVal)

    print("Precision: {:.5f}".format(precision))
    print("Recall:    {:.5f}".format(recall))
    print("F1 Score:  {:.5f}".format(fScore))
    print("MAE:       {:.5f}".format(mae))
    print("MSE:       {:.5f}".format(mse))
    print("RMSE:      {:.5f}".format(rmse))

    print()

def PrintPredictionResults():
    global testedSurvivors
    global testedNonSurvivors
    global lastAppliedModel
    userInput = 0
    resultsOptions = [
        "1) Survivors",
        "2) Non Survivors",
        "E) Exit"
    ]
    os.system(clearCMD)
    if len(testedSurvivors) == 0 or len(testedNonSurvivors) == 0:
        print("No data detected! Apply a model first.")
        return
    while userInput != "E":
        print("Model Used:", lastAppliedModel)
        for options in resultsOptions:
            print(options)
        userInput = input("Predictions to list:").capitalize()
        if userInput == "1":
            while userInput != 'E':
                os.system(clearCMD)
                print("Survivors\n", testedSurvivors.to_string(), '\n')
                userInput = PredictionPlots(testedSurvivors)
            userInput = ''
        if userInput == "2":
            while userInput != 'E':
                os.system(clearCMD)
                print("Non Survivors\n", testedNonSurvivors.to_string(), '\n')
                userInput = PredictionPlots(testedNonSurvivors)
            userInput = ''
        os.system(clearCMD)

def VisualizeEda(data, visualizeInput):
    category = "None"
    isNaDataList = ["Passenger Fare", "Ticket Class", "Ticket Number", "Cabin", "Age", "Gender", "NumSiblingSpouse", "NumParentChild", "Survived"]
    corrDataList = ["Passenger Fare", "Ticket Class", "Age", "Gender", "NumSiblingSpouse", "NumParentChild", "Survived"]
    categorizedDataList = ["Ticket Class", "C", "Gender", "NumSiblingSpouse", "NumParentChild"]
    if visualizeInput == "1":
        category = "Ticket Class"
    elif visualizeInput == "2":
        category = "C"
    elif visualizeInput == "3":
        category = "Gender"
    elif visualizeInput == "4":
        category = "NumSiblingSpouse"
    elif visualizeInput == "5":
        category = "NumParentChild"
    if category in categorizedDataList:
        plt = data[[category, 'Survived']].groupby(category).mean().Survived.plot(kind='bar')
        plt.set_xlabel(category)
        plt.set_ylabel('Survival Probability')
        matplotlib.pyplot.show()
    if visualizeInput == "6":
        for isNaData in isNaDataList:
            validDataPercentage = data[isNaData].isnull().sum() / len(data.index)
            print("Percentage of NaN in", isNaData, ":", validDataPercentage*100,"%")
            if isNaData == "Cabin":
                embarkationIsnaSum = len(data.index)-(data['C'].sum()+data['Q'].sum()+data['S'].sum())
                print("Percentage of NaN in Embarkation Country : ", (embarkationIsnaSum/len(data.index))*100)
    if visualizeInput == "7":
        corr_matrix = data[corrDataList].corr()
        matplotlib.pyplot.figure(figsize=(9, 8))
        sns.heatmap(data = corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)
        matplotlib.pyplot.show()
    if visualizeInput == "0":
        # x = 0
        # survivedQ = 0
        # survivedC = 0
        # survivedS = 0
        # while x < len(data):
        #     if data.loc[x, 'Survived']:
        #         if data.loc[x, 'Q']:
        #             survivedQ += 1
        #         if data.loc[x, 'C']:
        #             survivedC += 1
        #         if data.loc[x, 'S']:
        #             survivedS += 1
        #     x += 1
        survivedQ = data[(data['Survived'] == 1) & (data['Q'] == 1)]['Q'].sum()
        survivedC = data[(data['Survived'] == 1) & (data['C'] == 1)]['C'].sum()
        survivedS = data[(data['Survived'] == 1) & (data['S'] == 1)]['S'].sum()
        totalQ = data['Q'].sum()
        totalC = data['C'].sum()
        totalS = data['S'].sum()
        survivedPctgQ = survivedQ / totalQ if totalQ != 0 else 0
        survivedPctgC = survivedC / totalC if totalC != 0 else 0
        survivedPctgS = survivedS / totalS if totalS != 0 else 0
        tmpdata = {
            'survivedQ': [survivedPctgQ],
            'survivedC': [survivedPctgC],
            'survivedS': [survivedPctgS]
        }
        tmpdf = pandas.DataFrame(tmpdata)
        plt = tmpdf[['survivedQ','survivedC','survivedS']].plot(kind='bar',edgecolor='white')
        plt.set_xticks([])
        plt.set_xticklabels([])
        plt.set_xlabel('Embarkation Country')
        plt.set_ylabel('Survival Probability')
        matplotlib.pyplot.show()

# dataPath = ""
# Menu()
# rawData, rawTest = Setup()
# # rawData = pandas.read_csv(dataPath + "/train/MS_1_Scenario_train.csv")
# # test = pandas.read_csv(dataPath + "/test/MS_1_Scenario_test.csv")

# rawData = eda.Clean(rawData)
# test = eda.Clean(rawTest)

# extractedData = eda.Extract(rawData)
# test = eda.Extract(test)

# testedSurvivors = pandas.DataFrame(columns=test.columns)
# testedNonSurvivors = pandas.DataFrame(columns=test.columns)

# lastAppliedModel = ''

# userInput = 0
# while Operations():
#     os.system(clearCMD)
#     if userInput == "1":
#         pass
#     if userInput == "2":
#         print("Input Training Data")
#         print(rawData.to_string(), "\n")
#     if userInput == "3":
#         print("Cleaned Training Data")
#         print(extractedData.to_string(), "\n")
#     if userInput == "4":
#         print("Cleaned Testing Data")
#         print(test.to_string(), "\n")
#     if userInput == "5":
#         os.system(clearCMD)
#         print("1) Original data")
#         print("2) Extracted data")
#         print("3) Test data")
#         userInput = input("Data to be analysed:")
#         if userInput == "1":
#             data = rawData
#         elif userInput == "2":
#             data = extractedData
#         else:
#             data = test
#         print()
#         while Plots(data):
#             pass
#     if userInput == "6":
#         FilteredTable()
#     if userInput == "7":
#         LogisticRegression()
#     if userInput == "8":
#         KNearestNeigbour()
#     if userInput == "9":
#         PrintPredictionResults()