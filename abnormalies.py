import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

output_cleaned_file = 'data/cleaned_data.csv'
output_abnormalities_file = 'data/abnormal_data.csv'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

# Load the CSV file into a DataFrame
df = pd.read_csv('data/cabin/cleaned_cabin.csv')

def Operations():
    global column_name
    operationList = [
        "1) Passenger ID",
        "2) Passenger Fare",
        "3) Ticket Class",
        "4) Ticket Number",
        "5) Cabin",
        "6) Age",
        "7) Gender",
        "8) NumSiblings",
        "9) NumParentChild",
        "Press E to exit"
    ]

    for operation in operationList:
        print(operation)
    
    choice = input("Choose a column to find abnormalities or E to exit:")

    if choice.capitalize() == "E":
        return False
    else:
        column_name = get_column_name(choice)
        return True

def get_column_name(choice):
    column_mapping = {
        "1": 'Passenger ID',
        "2": 'Passenger Fare',
        "3": 'Ticket Class',
        "4": 'Ticket Number',
        "5": 'Cabin',
        "6": 'Age',
        "7": 'Gender',
        "8": 'NumSiblingSpouse',
        "9": 'NumParentChild',
    }
    return column_mapping.get(choice)

def Conversion(column_name):
    if column_name is None:
        return  # No valid column selected

    # Reset the DataFrame to the original data
    df = pd.read_csv('data/cabin/cleaned_cabin.csv')

    # Initialize a LabelEncoder
    label_encoder = LabelEncoder()

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
