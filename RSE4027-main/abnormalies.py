import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

output_cleaned_file = 'cleaned_data.csv'
output_abnormalities_file = 'abnormal_data.csv'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Load the CSV file into a DataFrame
df = pd.read_csv('./modified_data.csv')

def Operations():
    global column_name
    operationList = [
        "1) Patient ID",
        "2) Gender",
        "3) Age",
        "4) Height",
        "5) Weight",
        "6) fam_hist_over-wt",
        "7) FAVC",
        "8) FCVC",
        "9) NCP",
        "10) CAEC",
        "11) SMOKE",
        "12) CH20",
        "13) SCC",
        "14) FAF",
        "15) TUE",
        "16) CALC",
        "17) MTRANS",
        "18) Obesity_Level"

    ]

    for operation in operationList:
        print(operation)
    
    choice = input("Choose a column to find abnormalities:")

    if choice.capitalize() == "E":
        return False
    else:
        column_name = get_column_name(choice)
        return True

def get_column_name(choice):
    column_mapping = {
        "1": 'Patient ID',
        "2": 'Gender',
        "3": 'Age',
        "4": 'Height',
        "5": 'Weight',
        "6": 'fam_hist_over-wt',
        "7": 'FAVC',
        "8": 'FCVC',
        "9": 'NCP',
        "10": 'CAEC',
        "11": 'SMOKE',
        "12": 'CH20',
        "13": 'SCC',
        "14": 'FAF',
        "15": 'TUE',
        "16": 'CALC',
        "17": 'MTRANS',
        "18": 'Obesity_Level'
    }
    return column_mapping.get(choice)


def Conversion(column_name):
    if column_name is None:
        return  # No valid column selected

    # Reset the DataFrame to the original data
    df = pd.read_csv('./Team5-MS_2/MS_2_Scenario_data.csv')

    # Initialize a LabelEncoder
    label_encoder = LabelEncoder() # One hot encoding | pd.get_dummies

    # Convert alphanumeric values to numerical values
    df[column_name] = label_encoder.fit_transform(df[column_name])

    # Save the modified data to a new CSV file
    df.to_csv(output_cleaned_file, index=False)

    # Load the modified data from the new CSV file
    modified_df = pd.read_csv(output_cleaned_file)
    isolation_forest(column_name, modified_df)

def isolation_forest(column_name, modified_df):
    # Select the column where you want to detect abnormalities
    selected_column = modified_df[column_name]

    # Create an Isolation Forest model
    model = IsolationForest(contamination=0.05)  # Adjust the contamination parameter as needed

    # Fit the model to the selected column
    model.fit(selected_column.values.reshape(-1, 1))

    # Predict outliers using the Isolation Forest model
    outlier_predictions = model.predict(selected_column.values.reshape(-1, 1))

    # Extract and print the rows with outliers
    outlier_rows = modified_df[outlier_predictions == -1]
    print("Abnormal rows:")
    print(outlier_rows)

    # Write the abnormal data to a new CSV file
    outlier_rows.to_csv(output_abnormalities_file, index=False)

while Operations():
    Conversion(column_name)
