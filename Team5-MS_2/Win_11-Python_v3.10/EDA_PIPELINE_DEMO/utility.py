## 
# Changelog:
# - 17/11/23
#   Created utility library
##

# Dataframe Import
import pandas

# Math Import
import numpy

# Metrics Import
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef, mean_absolute_error, mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score

# Graphing Import
import matplotlib.pyplot as plt

def Str2NaN(value):
    if value == "0":
        value = numpy.nan
    return value

def Metrics(predictionData, predictionValue, config):
    trueValue = predictionData[config['Parameters']['Prediction Element']].values
    predictedValue = predictionValue

    predictedData = pandas.DataFrame(data=predictedValue, columns=['Prediction'])
    predictedData = pandas.get_dummies(predictedData, columns=['Prediction'], drop_first=False, prefix='', prefix_sep='')

    average = None
    if config['Multi-Class']:
        auc         = roc_auc_score(trueValue, predictedData, multi_class='ovo')
        precision   = precision_score(trueValue, predictedValue, average=average, labels=list(config['Classifications'].values()))
        recall      = recall_score(trueValue, predictedValue, average=average, labels=list(config['Classifications'].values()))
        f1score     = f1_score(trueValue, predictedValue, average=average, labels=list(config['Classifications'].values()))
    else:
        auc = roc_auc_score(trueValue, predictedValue)
        precision   = precision_score(trueValue, predictedValue)
        recall      = recall_score(trueValue, predictedValue)
        f1score     = f1_score(trueValue, predictedValue)
    ca      = accuracy_score(trueValue, predictedValue)
    mcc     = matthews_corrcoef(trueValue, predictedValue)
    mae     = mean_absolute_error(trueValue, predictedValue)
    mse     = mean_squared_error(trueValue, predictedValue)
    rmse    = numpy.sqrt(mse)
    metrics = {
        'AUC'   : auc,
        'CA'    : ca,
        'MCC'   : mcc,
        'MAE'   : mae,
        'MSE'   : mse,
        'RMSE'  : rmse,
        'F1'    : f1score,
        'Precision' : precision,
        'Recall'    : recall
    }
    return metrics

def VisualizeMetrics(lastAppliedModel, metrics, config):
    metricLabel = []
    metricCount = []
    if config['Multi-Class']:
        i = 0
        while i < len(metrics['Precision']):
            metricLabel.append('Precision')
            metricLabel.append('Recall')
            metricLabel.append('F1 Score')
            metricCount.append(metrics['Precision'][i])
            metricCount.append(metrics['Recall'][i])
            metricCount.append(metrics['F1'][i])
            print("Precision of", list(config['Classifications'].keys())[i], ": {:.5f}".format(metrics['Precision'][i]))
            print("Recall of   ", list(config['Classifications'].keys())[i], ": {:.5f}".format(metrics['Recall'][i]))
            print("F1 Score of ", list(config['Classifications'].keys())[i], ": {:.5f}".format(metrics['F1'][i]))
            print()
            i += 1
    else:
        metricLabel.append('Precision')
        metricLabel.append('Recall')
        metricLabel.append('F1 Score')
        metricCount.append(metrics['Precision'])
        metricCount.append(metrics['Recall'])
        metricCount.append(metrics['F1'])
        print("Precision: {:.5f}".format(metrics['Precision']))
        print("Recall:    {:.5f}".format(metrics['Recall']))
        print("F1 Score:  {:.5f}".format(metrics['F1']))

    scoreLabel = ['Accuracy']
    scoreLabel.append('AUC')
    scoreLabel.append('MCC')
    scoreLabel.append('MAE')
    scoreLabel.append('MSE')
    scoreLabel.append('RMSE')
    scoreCount = [metrics['CA']]
    scoreCount.append(metrics['AUC'])
    scoreCount.append(metrics['MCC'])
    scoreCount.append(metrics['MAE'])
    scoreCount.append(metrics['MSE'])
    scoreCount.append(metrics['RMSE'])
    print("Accuracy: {:.5f}".format(metrics['CA']))
    print("AUC:      {:.5f}".format(metrics['AUC']))
    print("MCC:      {:.5f}".format(metrics['MCC']))
    print("MAE:      {:.5f}".format(metrics['MAE']))
    print("MSE:      {:.5f}".format(metrics['MSE']))
    print("RMSE:     {:.5f}".format(metrics['RMSE']))
    print()
    
    scoreFig, scoreAx = plt.subplots()
    scoreBar = scoreAx.bar(scoreLabel, scoreCount)
    scoreAx.bar_label(scoreBar, fmt='{:.2f}')
    scoreFig.suptitle(lastAppliedModel)

    params = {
    'fontsize': 12
    }

    if config['Multi-Class']:
        metricFig, metricAx = plt.subplots(1,len(config['Classifications']), sharey=True)
        i = 0
        j = 0
        while j < len(config['Classifications']):
            metricBar = metricAx[j].bar(metricLabel[i:i+3], metricCount[i:i+3])
            metricAx[j].set_title(list(config['Classifications'].keys())[j], fontdict = params)
            metricAx[j].bar_label(metricBar, fmt='{:.2f}')
            i += 3
            j += 1
    else:
        metricFig, metricAx = plt.subplots()
        metricBar = metricAx.bar(metricLabel, metricCount)
        metricAx.bar_label(metricBar, fmt='{:.2f}')
    metricFig.suptitle(lastAppliedModel)
    plt.show()

def LoadSave(config):
    metricsCsv = pandas.read_csv(config['Save Path'])
    metricsSet = []
    i = 0
    while i < len(metricsCsv):
        metrics = {
            'Model' : metricsCsv.loc[i,'Model'],
            'AUC'   : metricsCsv.loc[i, 'AUC'].astype(float),
            'CA'    : metricsCsv.loc[i, 'CA'].astype(float),
            'MCC'   : metricsCsv.loc[i, 'MCC'].astype(float),
            'MAE'   : metricsCsv.loc[i, 'MAE'].astype(float),
            'MSE'   : metricsCsv.loc[i, 'MSE'].astype(float),
            'RMSE'  : metricsCsv.loc[i, 'RMSE'].astype(float),
            'F1'    : [],
            'Precision' : [],
            'Recall'    : []
        }
        j = 0
        while j < len(config['Classifications']):
            metrics['F1'].append(metricsCsv.loc[i, list(config['Classifications'].keys())[j] + '_F1'].astype(float))
            metrics['Precision'].append(metricsCsv.loc[i, list(config['Classifications'].keys())[j] + '_Precision'].astype(float))
            metrics['Recall'].append(metricsCsv.loc[i, list(config['Classifications'].keys())[j] + '_Recall'].astype(float))
            j += 1
        metricsSet.append(metrics)
        i += 1

    return metricsSet

def SaveSetup(config):
    saveData = pandas.DataFrame(columns=['Model', 'AUC', 'CA', 'MCC', 'MAE', 'MSE', 'RMSE'])
    if config['Multi-Class']:
        for classification in list(config['Classifications'].keys()):
            saveData.insert(len(saveData.columns), classification + '_F1', 0.0)
            saveData.insert(len(saveData.columns), classification + '_Precision', 0.0)
            saveData.insert(len(saveData.columns), classification + '_Recall', 0.0)
    else:
        saveData.insert(len(saveData.columns), 'F1', 0.0)
        saveData.insert(len(saveData.columns), 'Precision', 0.0)
        saveData.insert(len(saveData.columns), 'Recall', 0.0)
    return saveData

def SaveData(saveData, filepath):
    saveData.to_csv(filepath)

def UpdateSaveData(lastAppliedModel, saveData, metrics, config):
    if lastAppliedModel == None:
        return
    saveData.loc[lastAppliedModel, 'Model']  = lastAppliedModel
    saveData.loc[lastAppliedModel, 'AUC']  = metrics['AUC']
    saveData.loc[lastAppliedModel, 'CA']   = metrics['CA']
    saveData.loc[lastAppliedModel, 'MCC']  = metrics['MCC']
    saveData.loc[lastAppliedModel, 'MAE']  = metrics['MAE']
    saveData.loc[lastAppliedModel, 'MSE']  = metrics['MSE']
    saveData.loc[lastAppliedModel, 'RMSE'] = metrics['RMSE']

    if config['Multi-Class']:
        i = 0
        for classification in list(config['Classifications'].keys()):
            saveData.loc[lastAppliedModel,classification + '_F1'] = metrics['F1'][i]
            saveData.loc[lastAppliedModel,classification + '_Precision'] = metrics['Precision'][i]
            saveData.loc[lastAppliedModel,classification + '_Recall'] = metrics['Recall'][i]
            i += 1
    else:
        saveData.insert(len(saveData.columns), 'F1', 0.0)
        saveData.insert(len(saveData.columns), 'Precision', 0.0)
        saveData.insert(len(saveData.columns), 'Recall', 0.0)

    return saveData

def SaveResults(predictionData, config):
    results = pandas.DataFrame(columns=['ID', 'Obesity Level', 'Prediction'])
    isOWI = predictionData['Prediction'] == config['Classifications']['Overweight_Level_I']
    isOWII = predictionData['Prediction'] == config['Classifications']['Overweight_Level_II']
    results = predictionData.loc[isOWI | isOWII, ['ID', 'Obesity_Level', 'Prediction']]
    overweightClassification = {
        0 : 'Insufficient_Weight',
        1 : 'Normal_Weight',
        2 : 'Overweight_Level_I',
        3 : 'Overweight_Level_II',
        4 : 'Obesity_Type_I',
        5 : 'Obesity_Type_II',
        6 : 'Obesity_Type_III'
    }
    results['Obesity_Level'] = results['Obesity_Level'].replace(overweightClassification, regex=True)
    results['Prediction'] = results['Prediction'].replace(overweightClassification, regex=True)
    results.to_csv(config['Result Path'])
