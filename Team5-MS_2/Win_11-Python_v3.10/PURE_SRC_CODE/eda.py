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

clearCMD = 'cls'

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
    data['BMI'] = data['BMI'].round(2)

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
    updatedCorrList = ['G', 'Age','BMI','GR','FAVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','Obesity_Level']
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
    corr_matrix = data[updatedCorrList].corr()
    plt.pyplot.figure(figsize=(9, 8))
    sns.heatmap(data = corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)
    plt.pyplot.show()

def ObeseProbability(data):
    userInput = ''
    category = ''
    tmpBins = ''
    binList = ['Age','H','W','FCVC','NCP','CH2O','FAF','TUE']
    ynList = ['FAVC','SMOKE','SCC']
    while userInput != 'E':
        # os.system("cls")
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
            "E) Return to Previous Menu"
        ]
        print("Number of obese vs total entries:", (data['Obese'] == 1).sum() , "/", len(data.index))
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
        if userInput == "3":
            category = 'H'
            tmpBins = 'Height Bins'
        if userInput == "4":
            category = 'W'
            tmpBins = 'Weight Bins'
        if userInput == "5":
            obeseGrNo = ((data['Obese'] == 1) & (data['GR'] == 0)).sum()
            obeseGrYes = data[(data['Obese'] == 1) & (data['GR'] == 1)]['GR'].sum()
            totalGrNo = (data['GR'] == 0).sum()
            totalGrYes = (data['GR'] == 1).sum()
            obesePctgGrNo = obeseGrNo / totalGrNo if totalGrNo != 0 else 0
            obesePctgGrYes = obeseGrYes / totalGrYes if totalGrYes != 0 else 0
            tmpdata = {
                'No': [obesePctgGrNo],
                'Yes': [obesePctgGrYes]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['No','Yes']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('Family with Overweight History')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
        if userInput == "6":
            category = 'FAVC'
            xlabel = 'High Calorie Consumption'
        if userInput == "7":
            category = 'FCVC'
            tmpBins = 'FCVC Bins'
        if userInput == "8":
            category = 'NCP'
            tmpBins = 'NCP Bins'
        if userInput == "9":
            obeseCaecNo = ((data['Obese'] == 1) & (data['CAEC_no'] == 1)).sum()
            obeseCaecSometimes = ((data['Obese'] == 1) & (data['CAEC_Sometimes'] == 1)).sum()
            obeseCaecFrequently = ((data['Obese'] == 1) & (data['CAEC_Frequently'] == 1)).sum()
            obeseCaecAlways = ((data['Obese'] == 1) & (data['CAEC_Always'] == 1)).sum()
            totalCaecNo = (data['CAEC_no'] == 1).sum()
            totalCaecSometimes = (data['CAEC_Sometimes'] == 1).sum()
            totalCaecFrequently = (data['CAEC_Frequently'] == 1).sum()
            totalCaecAlways = (data['CAEC_Always'] == 1).sum()
            obesePctgCaecNo = obeseCaecNo / totalCaecNo if totalCaecNo != 0 else 0
            obesePctgCaecSometimes = obeseCaecSometimes / totalCaecSometimes if totalCaecSometimes != 0 else 0
            obesePctgCaecFrequently = obeseCaecFrequently / totalCaecFrequently if totalCaecFrequently != 0 else 0
            obesePctgCaecAlways = obeseCaecAlways / totalCaecAlways if totalCaecAlways != 0 else 0
            tmpdata = {
                'No': [obesePctgCaecNo],
                'Sometimes': [obesePctgCaecSometimes],
                'Frequently': [obesePctgCaecFrequently],
                'Always': [obesePctgCaecAlways]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['No','Sometimes','Frequently','Always']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('Patient consumes additional food between meals')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
        if userInput == "10":
            category = 'SMOKE'
            xlabel = 'Smoke'
        if userInput == "11":
            category = 'CH2O'
            tmpBins = 'CH2O Bins'
        if userInput == "12":
            category = 'SCC'
            xlabel = 'Keeps Track of Personal Calorie Intake'
        if userInput == "13":
            category = 'FAF'
            tmpBins = 'FAF Bins'
        if userInput == "14":
            category = 'TUE'
            tmpBins = 'TUE Bins'
        if userInput == "15":
            obeseCalcNo = ((data['Obese'] == 1) & (data['CALC_no'] == 1)).sum()
            obeseCalcSometimes = ((data['Obese'] == 1) & (data['CALC_Sometimes'] == 1)).sum()
            obeseCalcFrequently = ((data['Obese'] == 1) & (data['CALC_Frequently'] == 1)).sum()
            totalCalcNo = (data['CALC_no'] == 1).sum()
            totalCalcSometimes = (data['CALC_Sometimes'] == 1).sum()
            totalCalcFrequently = (data['CALC_Frequently'] == 1).sum()
            obesePctgCalcNo = obeseCalcNo / totalCalcNo if totalCalcNo != 0 else 0
            obesePctgCalcSometimes = obeseCalcSometimes / totalCalcSometimes if totalCalcSometimes != 0 else 0
            obesePctgCalcFrequently = obeseCalcFrequently / totalCalcFrequently if totalCalcFrequently != 0 else 0
            tmpdata = {
                'No': [obesePctgCalcNo],
                'Sometimes': [obesePctgCalcSometimes],
                'Frequently': [obesePctgCalcFrequently]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['No','Sometimes','Frequently']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('Patient consumes alcohol')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
        if userInput == "16":
            obeseMtransAutomobile = ((data['Obese'] == 1) & (data['Automobile'] == 1)).sum()
            obeseMtransMotorbike = ((data['Obese'] == 1) & (data['Motorbike'] == 1)).sum()
            obeseMtransPublic = ((data['Obese'] == 1) & (data['Public_Transportation'] == 1)).sum()
            obeseMtransBike = ((data['Obese'] == 1) & (data['Bike'] == 1)).sum()
            obeseMtransWalk = ((data['Obese'] == 1) & (data['Walking'] == 1)).sum()
            totalMtransAutomobile = (data['Automobile'] == 1).sum()
            totalMtransMotorbike = (data['Motorbike'] == 1).sum()
            totalMtransPublic = (data['Public_Transportation'] == 1).sum()
            totalMtransBike = (data['Bike'] == 1).sum()
            totalMtransWalk = (data['Walking'] == 1).sum()
            obesePctgMtransAutomobile = obeseMtransAutomobile / totalMtransAutomobile if totalMtransAutomobile != 0 else 0
            obesePctgMtransMotorbike = obeseMtransMotorbike / totalMtransMotorbike if totalMtransMotorbike != 0 else 0
            obesePctgMtransPublic = obeseMtransPublic / totalMtransPublic if totalMtransPublic != 0 else 0
            obesePctgMtransBike = obeseMtransBike / totalMtransBike if totalMtransBike != 0 else 0
            obesePctgMtransWalk = obeseMtransWalk / totalMtransWalk if totalMtransWalk != 0 else 0
            tmpdata = {
                'Automobile': [obesePctgMtransAutomobile],
                'Motorbike': [obesePctgMtransMotorbike],
                'Public Transportation': [obesePctgMtransPublic],
                'Bike': [obesePctgMtransBike],
                'Walking': [obesePctgMtransWalk]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['Automobile','Motorbike','Public Transportation','Bike','Walking']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel('Patient\'s Mode of Travelling')
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
        if category in binList:
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
            category = ''
        if category in ynList:
            obeseCategoryNo = ((data['Obese'] == 1) & (data[category] == 0)).sum()
            obeseCategoryYes = data[(data['Obese'] == 1) & (data[category] == 1)][category].sum()
            totalCategoryNo = (data[category] == 0).sum()
            totalCategoryYes = (data[category] == 1).sum()
            obesePctgCategoryNo = obeseCategoryNo / totalCategoryNo if totalCategoryNo != 0 else 0
            obesePctgCategoryYes = obeseCategoryYes / totalCategoryYes if totalCategoryYes != 0 else 0
            tmpdata = {
                'No': [obesePctgCategoryNo],
                'Yes': [obesePctgCategoryYes]
            }
            tmpdf = pandas.DataFrame(tmpdata)
            plt = tmpdf[['No','Yes']].plot(kind='bar',edgecolor='white')
            plt.set_xticks([])
            plt.set_xticklabels([])
            plt.set_xlabel(xlabel)
            plt.set_ylabel('Obese Probability')
            matplotlib.pyplot.show()
            category = ''
        os.system(clearCMD)
        
