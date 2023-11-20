## 
# Changelog:
# - 16/11/23
#   Reused cleaning agent for config['Binary'] columns
##

from sklearn.metrics import mean_absolute_error, mean_squared_error
from utility import Str2NaN
import numpy
import pandas
import matplotlib
import matplotlib as plt
import seaborn as sns
import os
import numpy as np


def Find(data, category):
    distributionTable = {}
    for entry in data.dict.get(category):
        if entry in distributionTable.keys():
            distributionTable[entry] += 1
        else:
            distributionTable[entry] = 1
    return distributionTable

def Clean(data, config):
    # Rename Columns
    data.rename(columns={'Patient ID' : 'ID'}, inplace=True)
    data.rename(columns={'Gender' : 'G'}, inplace=True)
    data.rename(columns={'Height' : 'H'}, inplace=True)
    data.rename(columns={'Weight' : 'W'}, inplace=True)
    data.rename(columns={'fam_hist_over-wt' : 'GR'}, inplace=True)

    # Abnormal
    data.insert(len(data.columns), "Abnormal", False)
    
    # Gender [G] (Consider Discretization), 'F' = female, 'M' = male, 'O' = others
    data = pandas.get_dummies(data, columns=['G'], prefix='', drop_first=False)
    data.columns = data.columns.str.lstrip('_')

    # Age [Age] (Consider Binning)
    data['Age'] = data['Age'].round(decimals=0).astype(int)

    # Height [H]
    data['H'] = data['H'].round(decimals=2)

    # Weight [W]
    data['W'] = data['W'].round(decimals=2)

    # Extra for analysis, could consider dropping H and W after [BMI] 
    data['BMI'] = data['W'] / (data['H'] * data['H'])

    # Family history of over-weight / Genetic Risk [GR]
    data['GR'] = data['GR'].str.lower()
    data['GR'] = data['GR'].replace(config['Binary'], regex=True)

    # High Caloric Intake [FAVC]
    data['FAVC'] = data['FAVC'].str.lower()
    data['FAVC'] = data['FAVC'].replace(config['Binary'], regex=True)

    # Vegetable Intake Frequency [FCVC]
    data['FCVC'] = data['FCVC'].round(decimals=2)

    # Main Meals [NCP]
    data['NCP'] = data['NCP'].round(decimals=0).astype(int)

    # In-between Meals [CAEC] (Consider Discretization)
    data = pandas.get_dummies(data, columns=['CAEC'], drop_first=False)

    # Smoker [SMOKE]
    data['SMOKE'] = data['SMOKE'].str.lower()
    data['SMOKE'] = data['SMOKE'].replace(config['Binary'], regex=True)

    # Water intake frequency [CH2O]
    data['CH2O'] = data['CH2O'].round(decimals=2)

    # Tracks calorie intake [SCC]
    data['SCC'] = data['SCC'].str.lower()
    data['SCC'] = data['SCC'].replace(config['Binary'], regex=True)

    # Physical Activity Frequency [PAF]
    data['FAF'] = data['FAF'].round(decimals=2)

    # Time spent using Technology [TUE]
    data['TUE'] = data['TUE'].round(decimals=2)

    # Consumes Alcohol [CALC] (In Text format) (Consider Discretization)
    data = pandas.get_dummies(data, columns=['CALC'], drop_first=False)

    # Mode of Travel [MTRANS] (In Text format) (Consider Discretization / Binning)
    data = pandas.get_dummies(data, columns=['MTRANS'], prefix='', drop_first=False)
    data.columns = data.columns.str.lstrip('_')

    # Obesity Level [Obesity_Level] (In Text format) (Classification)
    data['Obesity_Level'] = data['Obesity_Level'].replace(config['Classifications'], regex=True)
    data.insert(len(data.columns), 'Obese', 0)
    data.loc[data['Obesity_Level'] <= 1, 'Obese'] = 0
    data.loc[data['Obesity_Level'] > 1, 'Obese'] = 1

    # Drop Unamed Columns
    data.drop('Unnamed: 18', axis='columns', inplace=True)
    return data

def DropAbnormalities(data):
    normalData = pandas.DataFrame(columns=data.columns)
    i = 0
    while i < len(data):
        if data.loc[i, 'Abnormal'] == False:
            normalData.loc[len(normalData.index)] = data.loc[i]
        i += 1
    normalData.drop('Abnormal', axis='columns', inplace=True)
    return normalData

def NaEntries(data):
    naList = ['Age','H','W','GR','FAVC','NCP','SMOKE','CH2O','SCC','FAF','TUE','Obesity_Level']
    totalEntries = len(data.index)
    missingGenderEntries = totalEntries-((data['Female']==1).sum()+(data['Male']==1).sum())
    missingCaecEntries = totalEntries-((data['CAEC_no']==1).sum()+(data['CAEC_Sometimes']==1).sum()+(data['CAEC_Frequently']==1).sum()+(data['CAEC_Always']==1).sum())
    missingCalcEntries = totalEntries-((data['CALC_no']==1).sum()+(data['CALC_Sometimes']==1).sum()+(data['CALC_Frequently']==1).sum())
    missingMtransEntries = totalEntries-((data['Automobile']==1).sum()+(data['Motorbike']==1).sum()+(data['Public_Transportation']==1).sum()+(data['Bike']==1).sum()+(data['Walking']==1).sum())
    for dataColumn in naList:
        validDataPercentage = data[dataColumn].isnull().sum() / totalEntries
        print("Number(s) of missing data in", dataColumn, ":", data[dataColumn].isnull().sum(), "/", totalEntries, "(", validDataPercentage*100,"% )")
    print("Number(s) of missing data in Gender:", missingGenderEntries, "/", totalEntries, "(", missingGenderEntries/totalEntries,"% )")
    print("Number(s) of missing data in CAEC:", missingCaecEntries, "/", totalEntries, "(", missingCaecEntries/totalEntries,"% )")
    print("Number(s) of missing data in CALC:", missingCalcEntries, "/", totalEntries, "(", missingCalcEntries/totalEntries,"% )")
    print("Number(s) of missing data in MTRANS:", missingMtransEntries, "/", totalEntries, "(", missingMtransEntries/totalEntries,"% )")

def CorrelationMatrix(data):
    corrList = ['G','Age','H','W','GR','FAVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','Obesity_Level']
    updatedCorrList = ['Age','BMI','GR','FAVC','CAEC','CH2O','SCC','FAF','TUE','CALC','Obesity_Level'] #Only corr with > +/-0.1
    ## For screenshot analysis
    hwList = ['H','W','Obesity_Level']
    bmiList = ['BMI','Obesity_Level']
    # Prepare Gender for correlation matrix
    data.loc[data['Male'] == True, 'G'] = 0
    data.loc[data['Female'] == True, 'G'] = 1
    ## Prepare CAEC for correlation matrix
    data.loc[data['CAEC_no'] == True, 'CAEC'] = 0
    data.loc[data['CAEC_Sometimes'] == True, 'CAEC'] = 1
    data.loc[data['CAEC_Frequently'] == True, 'CAEC'] = 2
    data.loc[data['CAEC_Always'] == True, 'CAEC'] = 3
    ## Prepare CALC for correlation matrix, has no 'Always' in the entry
    data.loc[data['CALC_no'] == True, 'CALC'] = 0
    data.loc[data['CALC_Sometimes'] == True, 'CALC'] = 1
    data.loc[data['CALC_Frequently'] == True, 'CALC'] = 2
    # data.loc[data['CALC_Always'] == True, 'CALC'] = 3
    ## Prepare MTRANS for correlation matrix
    data.loc[data['Automobile'] == True, 'MTRANS'] = 0
    data.loc[data['Motorbike'] == True, 'MTRANS'] = 1
    data.loc[data['Public_Transportation'] == True, 'MTRANS'] = 2
    data.loc[data['Bike'] == True, 'MTRANS'] = 3
    data.loc[data['Walking'] == True, 'MTRANS'] = 4
    ## Change between corrList,updatedCorrList,bmiList,hwList to change correlation matrix selections
    corr_matrix = data[bmiList].corr()
    plt.pyplot.figure(figsize=(9, 8))
    sns.heatmap(data = corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)
    plt.pyplot.show()

def ObeseProbability(data):
    userInput = ''
    tmpBins = ''
    while userInput != 'E':
        os.system("cls")
        plotList = [
            "1) Gender vs Obese",
            "2) Age vs Obese",
            "3) Height vs Obese",
            "4) Weight vs Obese",
            "5) Family History Overweight vs Obese",
            "6) High-calorie Frequency vs Obese",
            "7) FCVC vs Obese",
            "8) NCP vs Obese",
            "9) CAEC vs Obese",
            "10) Smoke vs Obese",
            "11) CH2O vs Obese",
            "12) SCC vs Obese",
            "13) FAF vs Obese",
            "14) TUE vs Obese",
            "15) CALC vs Obese",
            "16) Mode of Transport vs Obese",
            "E) Exit Program"
        ]
        for plot in plotList:
            print(plot)
        userInput = input("Plot:").capitalize()

        if userInput == "1":
            obeseFemale = data[(data['Obese'] == 1) & (data['Female'] == 1)]['Female'].sum()
            obeseMale = data[(data['Obese'] == 1) & (data['Male'] == 1)]['Male'].sum()
            totalFemale = data['Female'].sum()
            totalMale = data['Male'].sum()
            obesePctgFemale = obeseFemale / totalFemale if totalFemale != 0 else 0
            obesePctgMale = obeseMale / totalMale if totalMale != 0 else 0
            tmpdata = {
                'Female': [obesePctgFemale],
                'Male': [obesePctgMale]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['Female','Male']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('Gender')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
        if userInput == "2":
            category = 'Age'
            tmpBins = 'Age Bins'
            # ageMin = data['Age'].min()
            # ageMax = data['Age'].max()
            # ageBins = np.linspace(ageMin, ageMax, num=6)
            # data['Age Bins'] = pandas.cut(data['Age'], bins=ageBins, include_lowest=True)
            # ageBinsEncoded = pandas.get_dummies(data['Age Bins'], prefix='Age')
            # data = pandas.concat([data, ageBinsEncoded], axis=1)
            # obeseRates = data.groupby('Age Bins')['Obese'].mean()
            # matplotlib.pyplot.figure(figsize=(8, 6))
            # obeseRates.plot(kind='bar', color='skyblue')
            matplotlib.pyplot.title('Obese Probability by Age Group')
            matplotlib.pyplot.xlabel('Age Group')
            # matplotlib.pyplot.ylabel('Obese Probability')
            # matplotlib.pyplot.xticks(rotation=45)  # Rotate x-axis labels for better readability
            # matplotlib.pyplot.grid(axis='y', linestyle='--', alpha=0.7)
            # matplotlib.pyplot.tight_layout()
            # matplotlib.pyplot.show()
        if userInput == "3":
            category = 'H'
            tmpBins = 'H Bins'
            # heightMin = data['H'].min()
            # heightMax = data['H'].max()
            # heightBins = np.linspace(heightMin, heightMax, num=6)
            # data['H Bins'] = pandas.cut(data['H'], bins=heightBins, include_lowest=True)
            # heightBinsEncoded = pandas.get_dummies(data['H Bins'], prefix='H')
            # data = pandas.concat([data, heightBinsEncoded], axis=1)
            # obeseRates = data.groupby('H Bins')['Obese'].mean()
            # matplotlib.pyplot.figure(figsize=(8, 6))
            # obeseRates.plot(kind='bar', color='skyblue')
            matplotlib.pyplot.title('Obese Probability by Height Group')
            matplotlib.pyplot.xlabel('Height Group')
            # matplotlib.pyplot.ylabel('Obese Probability')
            # matplotlib.pyplot.xticks(rotation=45)  # Rotate x-axis labels for better readability
            # matplotlib.pyplot.grid(axis='y', linestyle='--', alpha=0.7)
            # matplotlib.pyplot.tight_layout()
            # matplotlib.pyplot.show()
        if userInput == "5":
            obeseNoGr = ((data['Obese'] == 1) & (data['GR'] == 0)).sum()
            obeseHasGr = data[(data['Obese'] == 1) & (data['GR'] == 1)]['GR'].sum()
            totalNoGr = (data['GR'] == 0).sum()
            totalHasGr = (data['GR'] == 1).sum()
            obesePctgNoGr = obeseNoGr / totalNoGr if totalNoGr != 0 else 0
            obesePctgHasGr = obeseHasGr / totalHasGr if totalHasGr != 0 else 0
            tmpdata = {
                'No OW fam hist': [obesePctgNoGr],
                'Has OW fam hist': [obesePctgHasGr]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['No OW fam hist','Has OW fam hist']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('History')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
        if userInput == "6":
            obeseNoFavc = ((data['Obese'] == 1) & (data['FAVC'] == 0)).sum()
            obeseHasFavc = data[(data['Obese'] == 1) & (data['FAVC'] == 1)]['FAVC'].sum()
            totalNoFavc = (data['FAVC'] == 0).sum()
            totalHasFavc = (data['FAVC'] == 1).sum()
            obesePctgNoFavc = obeseNoFavc / totalNoFavc if totalNoFavc != 0 else 0
            obesePctgHasFavc = obeseHasFavc / totalHasFavc if totalHasFavc != 0 else 0
            tmpdata = {
                'Dont freq consumes high cal': [obesePctgNoFavc],
                'Freq consumes high cal': [obesePctgHasFavc]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['Dont freq consumes high cal','Freq consumes high cal']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('High Calorie Consumption')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
        if userInput == "9":
            obeseNoGr = ((data['Obese'] == 1) & (data['GR'] == 0)).sum()
            obeseHasGr = data[(data['Obese'] == 1) & (data['GR'] == 1)]['GR'].sum()
            totalNoGr = (data['GR'] == 0).sum()
            totalHasGr = (data['GR'] == 1).sum()
            obesePctgNoGr = obeseNoGr / totalNoGr if totalNoGr != 0 else 0
            obesePctgHasGr = obeseHasGr / totalHasGr if totalHasGr != 0 else 0
            tmpdata = {
                'No OW fam hist': [obesePctgNoGr],
                'Has OW fam hist': [obesePctgHasGr]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['No OW fam hist','Has OW fam hist']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('History')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()

        if userInput == "2" or userInput == "3":
            categoryMin = data[category].min()
            categoryMax = data[category].max()
            categoryBins = np.linspace(categoryMin, categoryMax, num=6)
            data[tmpBins] = pandas.cut(data[category], bins=categoryBins, include_lowest=True)
            categoryBinsEncoded = pandas.get_dummies(data[tmpBins], prefix=category)
            data = pandas.concat([data, categoryBinsEncoded], axis=1)
            obeseRates = data.groupby(tmpBins)['Obese'].mean()
            matplotlib.pyplot.figure(figsize=(8, 6))
            obeseRates.plot(kind='bar', color='skyblue')
            matplotlib.pyplot.ylabel('Obese Probability')
            matplotlib.pyplot.xticks(rotation=45)  # Rotate x-axis labels for better readability
            matplotlib.pyplot.grid(axis='y', linestyle='--', alpha=0.7)
            matplotlib.pyplot.tight_layout()
            matplotlib.pyplot.show()
        
