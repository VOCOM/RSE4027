import os

from setup import Setup
from utility import LoadSave, VisualizeMetrics

def ListModels(metricsSet):
    i = 0
    while i < len(metricsSet):
        print('{})'.format(i+1), metricsSet[i]['Model'])
        i += 1
    print('E) Exit')
    return input('Model to analyze:').capitalize()

config = Setup(configOnly=True)
clearCMD = config['Clear Command']

# Classifications
classification = {
    'Insufficient_Weight' : 0,
    'Normal_Weight' : 1,
    'Overweight_Level_II' : 3,
    'Overweight_Level_I' : 2,
    'Obesity_Type_III' : 6,
    'Obesity_Type_II' : 5,
    'Obesity_Type_I' : 4
}

config.update({'Classifications' : classification})
metricsSet = LoadSave(config)

userInput = None
while userInput != 'E':
    os.system(clearCMD)
    if userInput != "E" and userInput != None:
        VisualizeMetrics(metricsSet[int(userInput) - 1]['Model'], metricsSet[int(userInput) - 1], config)
    userInput = ListModels(metricsSet)


