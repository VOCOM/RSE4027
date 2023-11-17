## 
# @author Muhammad Syamim Bin Shamsulbani
# @brief Machine learning & Artificial Intelligence Program for RSE4207
# @version 1.0
# @date 16/11/23
#
# Changelog:
# - 16/11/23
#   Setup returns configuration parameters
##

# OS Import
import os

import pandas
import numpy as np
import sklearn.metrics
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import matplotlib.pyplot
import seaborn as sns

# Local Import
import eda

# CLI Clear Macro
clearCMD = "cls"

def Setup():
    configFile = open('../SCRIPTS_CFG/config.txt')
    isUnified = False
    splitRatio = 0
    for line in configFile.readlines():
        if "Unified dataset" in line:
            isUnified = bool(line.strip().split("\"")[1])
        if "SplitRatio (Train/Test): " in line:
            if isUnified:
                line = line.strip().split("\"")[1].split("/")
                splitRatio = float(line[0]) /100
        if "Training dataset: " in line:
            trainingDataPath = line.strip().split("\"")[1].replace("\\","/")
        if "Testing dataset: " in line:
            testDataPath = line.strip().split("\"")[1].replace("\\","/")

    trainData = pandas.read_csv(trainingDataPath)
    if isUnified:
        testData = trainData.copy()
        maxIndex = len(trainData)
        splitIndex = int(maxIndex * splitRatio)
        i = 0
        while i < splitIndex:
            testData.drop(axis='index', labels=i, inplace=True)
            i += 1
        while i < maxIndex:
            trainData.drop(axis='index', labels=i, inplace=True)
            i += 1
        testData.reset_index(inplace=True)
        trainData.reset_index(inplace=True)
        testData.drop(axis='columns', labels='index', inplace=True)
        trainData.drop(axis='columns', labels='index', inplace=True)
    else:
        testData = pandas.read_csv(testDataPath)
    
    config = [isUnified, splitRatio]

    return trainData, testData, config

def EDAOperations():
    userInput = ''
    operationList = [
        "1) Clear Screen",
        "2) Print original data table",
        "3) Print cleaned data table",
        "4) Print original test table",
        "5) Print cleaned test table",
        "6) Data Distributions",
        "7) EDA Visualization",
        "E) Exit Program"
    ]

    for operation in operationList:
        print(operation)
    userInput = input("Operation:").capitalize()

    return userInput

def MLOperations():
    userInput = ''
    operationList = [
        "1) Clear Screen",
        "2) Print extracted data table",
        "3) Print extracted test table",
        "4) Logistic Regression",
        "5) K-Nearest Neighbour",
        "6) Prediction Results",
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

def LogisticRegression(lastAppliedModel, testedSurvivors, testedNonSurvivors, extractedTrainData, extractedTestData, test=False):
    # Logistic Regression
    testedSurvivors.drop(testedSurvivors.index, inplace=True)
    testedNonSurvivors.drop(testedNonSurvivors.index, inplace=True)
    lastAppliedModel = 'Logistic Regression'
    regr = linear_model.LogisticRegression(max_iter=1000)
    inputParameters = [
        'Passenger Fare',
        'Ticket Class',
        'Age',
        'Gender',
        # 'NumParentChild',
        'NumSiblingSpouse',
        'Q',
        'C',
        'S'
    ]
    X = extractedTrainData[inputParameters].values
    y = list(extractedTrainData['Survived'])
    regr = regr.fit(X,y)

    predictions = []

    i = 0
    while i < len(extractedTestData):
        inFare = extractedTestData['Passenger Fare'][i]
        inClass = extractedTestData['Ticket Class'][i]
        inAge = extractedTestData['Age'][i]
        inGender = extractedTestData['Gender'][i]
        # inParent = extractedTestData['NumParentChild'][i]
        inSibling = extractedTestData['NumSiblingSpouse'][i]
        inQ = extractedTestData['Q'][i]
        inC = extractedTestData['C'][i]
        inS = extractedTestData['S'][i]
        predictedSurvival = regr.predict([[inFare, inClass, inAge, inGender, inSibling, inQ, inC, inS]])
        predictions.append(predictedSurvival)
        if predictedSurvival:
            testedSurvivors.loc[len(testedSurvivors.index)] = extractedTestData.loc[i]
        else:
            testedNonSurvivors.loc[len(testedNonSurvivors.index)] = extractedTestData.loc[i]
        i += 1
    
    predictions_rounded = np.round(predictions).astype(int)
    
    if not test:
        actualVal = list(extractedTestData['Survived'].values)
        mae, mse, rmse = eda.ErrorCalc(predictions_rounded, actualVal)
        mcc = matthews_corrcoef(list(extractedTestData['Survived']), predictions)
        AUC = roc_auc_score(list(extractedTestData['Survived']), predictions)
        label = ['Survived', 'Not Survived']
        cm = confusion_matrix(actualVal, predictions_rounded)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        disp.plot()
        plt.show()

        print("Logistic Regression Metrics")
        accuracy = accuracy_score(list(extractedTestData['Survived']), predictions)
        precision = precision_score(list(extractedTestData['Survived']), predictions)
        recall = recall_score(list(extractedTestData['Survived']), predictions)
        fScore = f1_score(list(extractedTestData['Survived']), predictions)
        print("Accuracy:  {:.5f}".format(accuracy))
        print("Precision: {:.5f}".format(precision))
        print("Recall:    {:.5f}".format(recall))
        print("F1 Score:  {:.5f}".format(fScore))
        print("AUC:       {:.5f}".format(AUC))
        print("MCC:       {:.5f}".format(mcc))
        print("MAE:       {:.5f}".format(mae))
        print("MSE:       {:.5f}".format(mse))
        print("RMSE:      {:.5f}".format(rmse))
        print()

    testedSurvivors.insert(len(testedSurvivors.columns),"To Insure", "Yes")
    testedNonSurvivors.insert(len(testedNonSurvivors.columns),"To Insure", "No")
    tmpData = pandas.concat([testedSurvivors,testedNonSurvivors])
    tmpData.to_csv("Outcome.csv")

    return lastAppliedModel, testedSurvivors, testedNonSurvivors

def FilteredTable():
    header = list(extractedData.keys())
    print(header)
    userInput = input("Choose a category to search for: ")
    if userInput in header:
        print(extractedData[userInput].to_string())
    else:
        os.system(clearCMD)
        print("Category not found\n")

def KNearestNeigbour(lastAppliedModel, testedSurvivors, testedNonSurvivors, extractedTrainData, extractedTestData, test=False):
    # K Nearest Neighbor
    testedSurvivors.drop(testedSurvivors.index, inplace=True)
    testedNonSurvivors.drop(testedNonSurvivors.index, inplace=True)
    lastAppliedModel = 'K-Nearest Neighbour'
    K = 200

    inputParameters = [
        'Passenger Fare',
        'Ticket Class',
        'Age',
        'Gender',
        # 'NumParentChild',
        'NumSiblingSpouse',
        'Q',
        'C',
        'S'
    ]
    X = extractedTrainData[inputParameters].values
    y = list(extractedTrainData['Survived'])
    
    knn_model = KNeighborsRegressor(n_neighbors = K)
    knn_model.fit(X, y)
    knn_model.feature_names_in_ = inputParameters
    
    predictions = knn_model.predict(extractedTestData[inputParameters])
    predictions_rounded = np.round(predictions).astype(int)
    
    i = 0
    for prediction in predictions_rounded:
        if prediction:
            testedSurvivors.loc[len(testedSurvivors.index)] = extractedTestData.loc[i]
        else:
            testedNonSurvivors.loc[len(testedNonSurvivors.index)] = extractedTestData.loc[i]
        i += 1

    if not test:
        actualVal = list(extractedTestData['Survived'].values)
        label = ['Survived', 'Not Survived']
        cm = confusion_matrix(actualVal, predictions_rounded)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        disp.plot()
        plt.show()

        print("KNN Metrics")
        accuracy = accuracy_score(list(extractedTestData['Survived']), predictions)
        precision = precision_score(actualVal, predictions_rounded)
        recall = recall_score(actualVal, predictions_rounded)
        fScore = f1_score(actualVal, predictions_rounded)
        mae, mse, rmse = eda.ErrorCalc(predictions_rounded, actualVal)
        mcc = matthews_corrcoef(actualVal, predictions_rounded)
        AUC = roc_auc_score(list(extractedTestData['Survived']), predictions)

        print("Accuracy:  {:.5f}".format(accuracy))
        print("Precision: {:.5f}".format(precision))
        print("Recall:    {:.5f}".format(recall))
        print("F1 Score:  {:.5f}".format(fScore))
        print("AUC:       {:.5f}".format(AUC))
        print("MCC:       {:.5f}".format(mcc))
        print("MAE:       {:.5f}".format(mae))
        print("MSE:       {:.5f}".format(mse))
        print("RMSE:      {:.5f}".format(rmse))

        print()

    return lastAppliedModel, testedSurvivors, testedNonSurvivors

def PrintPredictionResults(lastAppliedModel, testedSurvivors, testedNonSurvivors):
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
    categorizedDataList = ["Ticket Class", "Gender", "NumSiblingSpouse", "NumParentChild"]
    if visualizeInput == "1":
        category = "Ticket Class"
    elif visualizeInput == "2":
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
            'Q survived': [survivedPctgQ],
            'C survived': [survivedPctgC],
            'S survived': [survivedPctgS]
        }
        tmpdf = pandas.DataFrame(tmpdata)
        plt = tmpdf[['Q survived','C survived','S survived']].plot(kind='bar',edgecolor='white')
        plt.set_xticks([])
        plt.set_xticklabels([])
        plt.set_xlabel('Embarkation Country')
        plt.set_ylabel('Survival Probability')
        matplotlib.pyplot.show()
    elif visualizeInput == "3":
        category = "Gender"
        print("0 - Male")
        print("1 - Female")
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
        corr_matrix = data[corrDataList].corr()
        matplotlib.pyplot.figure(figsize=(9, 8))
        sns.heatmap(data = corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)
        matplotlib.pyplot.show()
    if visualizeInput == "7":
        for isNaData in isNaDataList:
            validDataPercentage = data[isNaData].isnull().sum() / len(data.index)
            print("Percentage of NaN in", isNaData, ":", validDataPercentage*100,"%")
            if isNaData == "Cabin":
                embarkationIsnaSum = len(data.index)-(data['C'].sum()+data['Q'].sum()+data['S'].sum())
                print("Percentage of NaN in Embarkation Country : ", (embarkationIsnaSum/len(data.index))*100)
        input("Press Enter key to continue...")
    os.system(clearCMD)
    # return False
