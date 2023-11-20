import pandas as pd

def round_columns_to_one_decimal(df, column_names):
    for column_name in column_names:
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in the DataFrame.")
            return df

        df[column_name] = df[column_name].round(1)

    print("Modified data:")
    print(df)

    return df

def round_columns_to_whole(df, column_names):
    for column_name in column_names:
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in the DataFrame.")
            return df

        df[column_name] = df[column_name].round(0)

    print("Modified data:")
    print(df)

    return df

# Load the CSV file into a DataFrame
df = pd.read_csv('./Team5-MS_2/MS_2_Scenario_data.csv')

# Choose the column for processing
exclude_decimal_column = "Age"  # Replace with the actual column name
column_name = "Age"  # Replace with the actual column name

# Filter rows with no decimal points in the 'Age' column
# filtered_df_age = df[df[column_name] % 1 == 0]

# Add a new column 'Gender' and convert 'Male' and 'Female' to binary 1 or 0
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Print the rows with no decimal points in the 'Age' column
#print("Rows with no decimal points in the 'Age' column:")
#print(filtered_df_age)

# Choose columns 'Height' and 'Weight' for rounding to 1 decimal place
columns_to_round = ['Height', 'Weight']

# Round selected columns to 1 decimal place
df[columns_to_round] = round_columns_to_one_decimal(df[columns_to_round], columns_to_round)

# Add a new column 'Gender' and convert 'Male' and 'Female' to binary 1 or 0
df['fam_hist_over-wt'] = df['fam_hist_over-wt'].map({'yes': 1, 'no': 0})

df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})

df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})

df['SCC'] = df['SCC'].map({'yes': 1, 'no': 0})

# Choose columns 'FCVC' and 'NCP' for rounding to WHOLE NUMBER
ROUNDING_WHOLE = ['Age','FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Round selected columns to 1 decimal place
df[ROUNDING_WHOLE] = round_columns_to_whole(df[ROUNDING_WHOLE], ROUNDING_WHOLE)

# Add a new column 'Gender' and convert 'Male' and 'Female' to binary 1 or 0
df['CAEC'] = df['CAEC'].map({'Frequently': 1, 'Sometimes': 0, 'no': -1})
df['CALC'] = df['CALC'].map({'Frequently': 1, 'Sometimes': 0, 'no': -1})
df['Obesity_Level'] = df['Obesity_Level'].map({'Obesity_Type_I':3, 'Obesity_Type_II':4, 'Obesity_Type_III':5,'Overweight_Level_II':2, 'Overweight_Level_I': 1, 'Normal_Weight': 0, 'Insufficient_Weight': -1})
# Save the modified columns to a single CSV file
output_modified_file = 'modified_data.csv'
df.loc[df[column_name] % 1 == 0, ['Patient ID','Gender','Age', 'Height', 'Weight', 'fam_hist_over-wt','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','Obesity_Level']].to_csv(output_modified_file, index=False)

# Print the modified DataFrame
print("Modified Data:")
print(df.loc[df[column_name] % 1 == 0, ['Patient ID','Gender','Age', 'Height', 'Weight','fam_hist_over-wt','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','Obesity_Level']])
