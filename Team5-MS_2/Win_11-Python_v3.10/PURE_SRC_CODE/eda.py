## 
# Changelog:
# - 16/11/23
#   Reused cleaning agent for binary columns
##

from sklearn.metrics import mean_absolute_error, mean_squared_error
from utility import Str2NaN
import numpy
import pandas

def Find(data, category):
    distributionTable = {}
    for entry in data.dict.get(category):
        if entry in distributionTable.keys():
            distributionTable[entry] += 1
        else:
            distributionTable[entry] = 1
    return distributionTable

def Clean(data):
    # Rename Columns
    data.rename(columns={'Patient ID' : 'ID'}, inplace=True)
    data.rename(columns={'Gender' : 'G'}, inplace=True)
    data.rename(columns={'Height' : 'H'}, inplace=True)
    data.rename(columns={'Weight' : 'W'}, inplace=True)
    data.rename(columns={'fam_hist_over-wt' : 'GR'}, inplace=True)

    # Abnormal
    data.insert(len(data.columns), "Abnormal", False)
    
    # Gender [G] (Consider Discretization)
    data['G'] = data['G'].str.lower()
    data['G'] = data['G'].replace({'female' : 1}, regex=True)
    data['G'] = data['G'].replace({'male' : 0}, regex=True)

    # Age [Age] (Consider Binning)
    data['Age'] = data['Age'].round(decimals=0)

    # Height [H]
    data['H'] = data['H'].round(decimals=2)

    # Weight [W]
    data['W'] = data['W'].round(decimals=2)

    # Family history of over-weight / Genetic Risk [GR]
    data['GR'] = data['GR'].str.lower()
    data['GR'] = data['GR'].replace({'yes' : 1}, regex=True)
    data['GR'] = data['GR'].replace({'no' : 0}, regex=True)

    # High Caloric Intake [FAVC]
    data['FAVC'] = data['FAVC'].str.lower()
    data['FAVC'] = data['FAVC'].replace({'yes' : 1}, regex=True)
    data['FAVC'] = data['FAVC'].replace({'no' : 0}, regex=True)

    # Vegetable Intake Frequency [FCVC]
    data['FCVC'] = data['FCVC'].round(decimals=2)

    # Main Meals [NCP]
    data['NCP'] = data['NCP'].round(decimals=0)

    # In-between Meals [CAEC] (Consider Discretization)

    # Smoker [SMOKE]
    data['SMOKE'] = data['SMOKE'].str.lower()
    data['SMOKE'] = data['SMOKE'].replace({'yes' : 1}, regex=True)
    data['SMOKE'] = data['SMOKE'].replace({'no' : 0}, regex=True)

    # Water intake frequency [CH2O]
    data['CH2O'] = data['CH2O'].round(decimals=2)

    # Tracks calorie intake [SCC]
    data['SCC'] = data['SCC'].str.lower()
    data['SCC'] = data['SCC'].replace({'yes' : 1}, regex=True)
    data['SCC'] = data['SCC'].replace({'no' : 0}, regex=True)

    # Physical Activity Frequency [PAF]
    data['FAF'] = data['FAF'].round(decimals=2)

    # Time spent using Technology [TUE]
    data['TUE'] = data['TUE'].round(decimals=2)

    # Consumes Alcohol [CALC] (In Text format) (Consider Discretization)

    # Mode of Travel [MTRANS] (In Text format) (Consider Discretization / Binning)

    # Obesity Level [Obesity_Level] (In Text format) (Classification)

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

def ErrorCalc(predicted, actual):
    mae = mean_absolute_error(predicted, actual)
    mse = mean_squared_error(predicted, actual)
    rmse = numpy.sqrt(mse)

    return mae, mse, rmse
