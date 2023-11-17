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
    # Classification
    classification = {
        'Insufficient_Weight' : 0,
        'Normal_Weight' : 1,
        'Overweight_Level_I' : 2,
        'Overweight_Level_II' : 3,
        'Obesity_Type_I' : 4,
        'Obesity_Type_II' : 5,
        'Obesity_Type_III' : 6
    }
    # Binary Discretisation
    binary = {
        'no' : 0,
        'yes' : 1
    }

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
    data['Age'] = data['Age'].round(decimals=0).astype(int)

    # Height [H]
    data['H'] = data['H'].round(decimals=2)

    # Weight [W]
    data['W'] = data['W'].round(decimals=2)

    # Family history of over-weight / Genetic Risk [GR]
    data['GR'] = data['GR'].str.lower()
    data['GR'] = data['GR'].replace(binary, regex=True)

    # High Caloric Intake [FAVC]
    data['FAVC'] = data['FAVC'].str.lower()
    data['FAVC'] = data['FAVC'].replace(binary, regex=True)

    # Vegetable Intake Frequency [FCVC]
    data['FCVC'] = data['FCVC'].round(decimals=2)

    # Main Meals [NCP]
    data['NCP'] = data['NCP'].round(decimals=0).astype(int)

    # In-between Meals [CAEC] (Consider Discretization)

    # Smoker [SMOKE]
    data['SMOKE'] = data['SMOKE'].str.lower()
    data['SMOKE'] = data['SMOKE'].replace(binary, regex=True)

    # Water intake frequency [CH2O]
    data['CH2O'] = data['CH2O'].round(decimals=2)

    # Tracks calorie intake [SCC]
    data['SCC'] = data['SCC'].str.lower()
    data['SCC'] = data['SCC'].replace(binary, regex=True)

    # Physical Activity Frequency [PAF]
    data['FAF'] = data['FAF'].round(decimals=2)

    # Time spent using Technology [TUE]
    data['TUE'] = data['TUE'].round(decimals=2)

    # Consumes Alcohol [CALC] (In Text format) (Consider Discretization)

    # Mode of Travel [MTRANS] (In Text format) (Consider Discretization / Binning)

    # Obesity Level [Obesity_Level] (In Text format) (Classification)
    data['Obesity_Level'] = data['Obesity_Level'].replace(classification, regex=True)

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
