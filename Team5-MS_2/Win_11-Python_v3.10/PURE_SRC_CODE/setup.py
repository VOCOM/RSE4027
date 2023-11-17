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

# Dataframe Import
import pandas

# Graphing Imports
import matplotlib.pyplot as plt
import seaborn as sns

# Math Import
import numpy

def Setup():
    configFile = open('../SCRIPTS_CFG/config.txt')
    clearCMD = ''
    isUnified = False
    splitRatio = 0
    maxIterations = 0
    kMeans = 0
    for line in configFile.readlines():
        if "Clear Command: " in line:
            clearCMD = line.strip().split("\"")[1]
        if "Unified dataset: " in line:
            isUnified = bool(line.strip().split("\"")[1])
        if "SplitRatio (Train/Test): " in line:
            if isUnified:
                line = line.strip().split("\"")[1].split("/")
                splitRatio = float(line[0]) /100
        if "Training dataset: " in line:
            trainingDataPath = line.strip().split("\"")[1].replace("\\","/")
        if "Testing dataset: " in line:
            testDataPath = line.strip().split("\"")[1].replace("\\","/")
        if "Max Iteration: " in line:
            maxIterations = int(line.strip().split("\"")[1])
        if "K-Means: " in line:
            kMeans = int(line.strip().split("\"")[1])

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
    
    config = {
        'Clear Command' : clearCMD,
        'Unified' : isUnified,
        'Split Ratio' : splitRatio,
        'Max Iteration' : maxIterations,
        'K-Means' : kMeans
    }

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
        bottom = numpy.zeros(3)
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
        bottom = numpy.zeros(3)
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
        bottom = numpy.zeros(2)
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
            'Survived' : numpy.zeros(maxVal+1, dtype=int),
            'Not Survived' : numpy.zeros(maxVal+1, dtype=int)
        }
        while x < len(data):
            if data.loc[x,'Survived']:
                VertDependantSurvival['Survived'][data.loc[x,'NumParentChild']]+=1
            else:
                VertDependantSurvival['Not Survived'][data.loc[x,'NumParentChild']]+=1
            x+=1
        bottom = numpy.zeros(maxVal+1)
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
            'Survived' : numpy.zeros(maxVal+1, dtype=int),
            'Not Survived' : numpy.zeros(maxVal+1, dtype=int)
        }
        while x < len(data):
            if data.loc[x,'Survived']:
                VertDependantSurvival['Survived'][data.loc[x,'NumSiblingSpouse']]+=1
            else:
                VertDependantSurvival['Not Survived'][data.loc[x,'NumSiblingSpouse']]+=1
            x+=1
        bottom = numpy.zeros(maxVal+1)
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

def FilteredTable():
    header = list(extractedData.keys())
    print(header)
    userInput = input("Choose a category to search for: ")
    if userInput in header:
        print(extractedData[userInput].to_string())
    else:
        os.system(clearCMD)
        print("Category not found\n")

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
