import os

from setup import Setup
from utility import LoadSave, VisualizeMetrics

def ListModels(metrics):
    i = 1
    for model in metrics['Model']:
        print('{})'.format(i), model)
        i += 1
    print('E) Exit')
    return input('Model to analyze:').capitalize()

config = Setup(configOnly=True)
metrics = LoadSave(config)
clearCMD = config['Clear Command']

userInput = None
while userInput != 'E':
    os.system(clearCMD)
    if userInput == "1":
        VisualizeMetrics(metrics['Model'][int(userInput)], metrics, config)
    userInput = ListModels(metrics)


