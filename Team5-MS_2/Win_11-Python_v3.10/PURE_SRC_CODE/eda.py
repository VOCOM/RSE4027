## 
# Changelog:
# - 16/11/23
#   Reused cleaning agent for config['Binary'] columns
##

from sklearn.metrics import mean_absolute_error, mean_squared_error
from utility import Str2NaN
import numpy
import pandas
import matplotlib as plt
import seaborn as sns


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
    data['Female'] = 0
    data['Male'] = 0
    data['G'] = data['G'].str.lower()
    data.loc[data['G'] == 'male', 'Male'] = 1
    data.loc[data['G'] == 'female', 'Female'] = 1
    # data['Others(Gender)'] = 0
    # data.loc[~data['G'].isin(['male','female']), 'Others(Gender)'] = 1
    data.drop('G', axis='columns', inplace=True)
    # data.loc[data['Male'] == '1','Gender'] = 0
    # data.loc[data['Female'] == '1','Gender'] = 1

    # Age [Age] (Consider Binning)
    data['Age'] = data['Age'].round(decimals=0).astype(int)

    # Height [H]
    data['H'] = data['H'].round(decimals=2)

    # Weight [W]
    data['W'] = data['W'].round(decimals=2)

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
    data['caecNo'] = 0
    data['caecSometimes'] = 0
    data['caecFrequently'] = 0
    data['caecAlways'] = 0
    data['CAEC'] = data['CAEC'].str.lower()
    data.loc[data['CAEC'] == 'no', 'caecNo'] = 1
    data.loc[data['CAEC'] == 'sometimes', 'Sometimes'] = 1
    data.loc[data['CAEC'] == 'frequently', 'Frequently'] = 1
    data.loc[data['CAEC'] == 'always', 'Always'] = 1
    # data['Others(CAEC)'] = 0
    # data.loc[~data['CAEC'].isin(['no','sometimes','frequently','always']), 'Others(CAEC)'] = 1
    data.drop('CAEC', axis='columns', inplace=True)

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

    # Mode of Travel [MTRANS] (In Text format) (Consider Discretization / Binning)

    # Obesity Level [Obesity_Level] (In Text format) (Classification)
    data['Obesity_Level'] = data['Obesity_Level'].replace(config['Classifications'], regex=True)

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

def VisualizeEda(data):
    data.loc[data['Male'] == '1','Gender'] = 0
    data.loc[data['Female'] == '1','Gender'] = 1
    # data.loc[data['Others(Gender)'] == '1','Gender'] = 2
    corr_matrix = data[['Gender','Age']].corr()
    plt.pyplot.figure(figsize=(9, 8))
    sns.heatmap(data = corr_matrix, cmap='BrBG', annot=True, linewidths=0.2)
    plt.pyplot.show()