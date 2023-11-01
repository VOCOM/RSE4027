import pandas as pd

# Load the CSV file into a DataFrame
csv_file = "Milestone1/train/MS_1_Scenario_train.csv"
df = pd.read_csv(csv_file)

# Define a function to filter rows with "0" in the "Cabin" column and save to a new CSV file
def preprocess_cabin_0(df):
    new_df = df[df['Cabin'] == '0']
    new_file = "data/cabin/filtered_cabin_0.csv"
    new_df.to_csv(new_file, index=False)

def preprocess_cabin_exclude(df):
    # Filter rows in the "Cabin" column that have alphabets followed by a numerical value
    filtered_df = df[df['Cabin'].str.contains(r'[A-Z][A-Z\W\s]|[^0-9]$')]
    # Save the filtered DataFrame to a new CSV file
    filtered_file = "data/cabin/cabin_excluded.csv"    
    filtered_df.to_csv(filtered_file, index=False)

def preprocess_cabin_cleaned(df):
    # Filter rows in the "Cabin" column based on your criteria
    filtered_df = df[~df['Cabin'].str.contains(r'[A-Z][A-Z\W\s]|[^0-9]$')]
    # Save the filtered DataFrame to a new CSV file
    filtered_file = "data/cabin/cleaned_cabin.csv"
    filtered_df.to_csv(filtered_file, index=False)

# Call the preprocessing function to filter and save rows
preprocess_cabin_0(df)
preprocess_cabin_exclude(df)
preprocess_cabin_cleaned(df)

