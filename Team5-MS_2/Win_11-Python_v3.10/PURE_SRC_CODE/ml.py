## 
# Changelog:
# - 17/11/23
#   Created machine learning library
##

# Math Import
import numpy

# Machine Learning Imports
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

def LogisticRegression(predictionData, trainData, testData, parameters, config):
    # Reset Prediction dataframe
    predictionData.drop(predictionData.index, inplace=True)
    # Logistic Regression
    regr = linear_model.LogisticRegression(max_iter=config['Max Iteration'])
    X = trainData[parameters['Input Parameters']].values
    y = list(trainData[parameters['Prediction Element']])
    regr = regr.fit(X,y)

    predictions = regr.predict(testData[parameters['Input Parameters']])
    predictionData = testData.copy()
    predictionData.insert(len(predictionData.columns), 'Prediction', predictions)
    predictionData.drop('Abnormal', axis='columns', inplace=True)
    print(predictionData.to_string())

    # i = 0
    # while i < len(testData):
    #     inFare = testData['Passenger Fare'][i]
    #     inClass = testData['Ticket Class'][i]
    #     inAge = testData['Age'][i]
    #     inGender = testData['Gender'][i]
    #     # inParent = testData['NumParentChild'][i]
    #     inSibling = testData['NumSiblingSpouse'][i]
    #     inQ = testData['Q'][i]
    #     inC = testData['C'][i]
    #     inS = testData['S'][i]
    #     predictedSurvival = regr.predict([[inFare, inClass, inAge, inGender, inSibling, inQ, inC, inS]])
    #     predictions.append(predictedSurvival)
    #     if predictedSurvival:
    #         positiveData.loc[len(positiveData.index)] = testData.loc[i]
    #     else:
    #         negativeData.loc[len(negativeData.index)] = testData.loc[i]
    #     i += 1
    
    # predictions_rounded = numpy.round(predictions).astype(int)
    
    # if not test:
    #     actualVal = list(testData['Survived'].values)
    #     mae, mse, rmse = eda.ErrorCalc(predictions_rounded, actualVal)
    #     mcc = matthews_corrcoef(list(testData['Survived']), predictions)
    #     AUC = roc_auc_score(list(testData['Survived']), predictions)
    #     label = ['Survived', 'Not Survived']
    #     cm = confusion_matrix(actualVal, predictions_rounded)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    #     disp.plot()
    #     plt.show()

    #     print("Logistic Regression Metrics")
    #     accuracy = accuracy_score(list(testData['Survived']), predictions)
    #     precision = precision_score(list(testData['Survived']), predictions)
    #     recall = recall_score(list(testData['Survived']), predictions)
    #     fScore = f1_score(list(testData['Survived']), predictions)
    #     print("Accuracy:  {:.5f}".format(accuracy))
    #     print("Precision: {:.5f}".format(precision))
    #     print("Recall:    {:.5f}".format(recall))
    #     print("F1 Score:  {:.5f}".format(fScore))
    #     print("AUC:       {:.5f}".format(AUC))
    #     print("MCC:       {:.5f}".format(mcc))
    #     print("MAE:       {:.5f}".format(mae))
    #     print("MSE:       {:.5f}".format(mse))
    #     print("RMSE:      {:.5f}".format(rmse))
    #     print()

    # positiveData.insert(len(positiveData.columns),"To Insure", "Yes")
    # negativeData.insert(len(negativeData.columns),"To Insure", "No")
    # tmpData = pandas.concat([positiveData,negativeData])
    # tmpData.to_csv("Outcome.csv")

    return 'Logistic Regression'


def KNearestNeigbour(predictionData, trainData, testData, parameters, lastAppliedModel):
    # K Nearest Neighbor
    testedSurvivors.drop(testedSurvivors.index, inplace=True)
    testedNonSurvivors.drop(testedNonSurvivors.index, inplace=True)
    lastAppliedModel = 'K-Nearest Neighbour'
    K = 200

    parameters = [
        'Passenger Fare',
        'Ticket Class',
        'Age',
        'Gender',
        # 'NumParentChild',
        'NumSiblingSpouse',
        'Q',
        'C',
        'S'
    ]
    X = extractedTrainData[parameters].values
    y = list(extractedTrainData['Survived'])
    
    knn_model = KNeighborsRegressor(n_neighbors = K)
    knn_model.fit(X, y)
    knn_model.feature_names_in_ = parameters
    
    predictions = knn_model.predict(extractedTestData[parameters])
    predictions_rounded = numpy.round(predictions).astype(int)
    
    i = 0
    for prediction in predictions_rounded:
        if prediction:
            testedSurvivors.loc[len(testedSurvivors.index)] = extractedTestData.loc[i]
        else:
            testedNonSurvivors.loc[len(testedNonSurvivors.index)] = extractedTestData.loc[i]
        i += 1

    if not test:
        actualVal = list(extractedTestData['Survived'].values)
        label = ['Survived', 'Not Survived']
        cm = confusion_matrix(actualVal, predictions_rounded)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
        disp.plot()
        plt.show()

        print("KNN Metrics")
        accuracy = accuracy_score(list(extractedTestData['Survived']), predictions)
        precision = precision_score(actualVal, predictions_rounded)
        recall = recall_score(actualVal, predictions_rounded)
        fScore = f1_score(actualVal, predictions_rounded)
        mae, mse, rmse = eda.ErrorCalc(predictions_rounded, actualVal)
        mcc = matthews_corrcoef(actualVal, predictions_rounded)
        AUC = roc_auc_score(list(extractedTestData['Survived']), predictions)

        print("Accuracy:  {:.5f}".format(accuracy))
        print("Precision: {:.5f}".format(precision))
        print("Recall:    {:.5f}".format(recall))
        print("F1 Score:  {:.5f}".format(fScore))
        print("AUC:       {:.5f}".format(AUC))
        print("MCC:       {:.5f}".format(mcc))
        print("MAE:       {:.5f}".format(mae))
        print("MSE:       {:.5f}".format(mse))
        print("RMSE:      {:.5f}".format(rmse))

        print()

    return lastAppliedModel, testedSurvivors, testedNonSurvivors
