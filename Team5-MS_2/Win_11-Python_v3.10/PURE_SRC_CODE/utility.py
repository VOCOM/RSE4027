## 
# Changelog:
# - 17/11/23
#   Created utility library
##

# Math Import
import numpy

# Metrics Import
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef, mean_absolute_error, mean_squared_error

# Graphing Import
import matplotlib.pyplot as plt

def Str2NaN(value):
    if value == "0":
        value = numpy.nan
    return value

def Metrics(testValue, predictionValue, predictedProbability = 0, isMultiClass = False):
    trueValue = testValue
    predictedValue = predictionValue
    average = 'micro'
    if isMultiClass:
        auc         = roc_auc_score(trueValue, predictedProbability, multi_class='ovr')
        precision   = precision_score(trueValue, predictedValue, average=average)
        recall      = recall_score(trueValue, predictedValue, average=average)
        f1score     = f1_score(trueValue, predictedValue, average=average)
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

def VisualizeMetrics(metrics):
    fig, ax = plt.subplots()
    label = metrics.keys()
    count = metrics.values()
    bar = ax.bar(label, count)
    ax.bar_label(bar, fmt='{:.2f}')
    plt.show()

    print("Accuracy:  {:.5f}".format(metrics['CA']))
    print("Precision: {:.5f}".format(metrics['Precision']))
    print("Recall:    {:.5f}".format(metrics['Recall']))
    print("F1 Score:  {:.5f}".format(metrics['F1']))
    print("AUC:       {:.5f}".format(metrics['AUC']))
    print("MCC:       {:.5f}".format(metrics['MCC']))
    print("MAE:       {:.5f}".format(metrics['MAE']))
    print("MSE:       {:.5f}".format(metrics['MSE']))
    print("RMSE:      {:.5f}".format(metrics['RMSE']))
    print()
    