from FileReader.DataTypes import DATA

import math
import re

def Find(data, category):
    distributionTable = {}
    for entry in data.dict.get(category):
        if entry in distributionTable.keys():
            distributionTable[entry] += 1
        else:
            distributionTable[entry] = 1
    return distributionTable

def Info(data):
    for header in data.dict.keys():
        print(header, len(data.dict.get(header)), "entries")
    print()
        
    for header in data.dict.keys():
        print(header)
        distributionTable = Find(data, header)
        PrintDistribution(distributionTable)

def PrintDistribution(distributionTable):
        for distribution in distributionTable.items():
            print(distribution)
        print()

def Clean(data):
    data["Gender"] = data["Gender"].replace({"female" : 1}, regex=True)
    data["Gender"] = data["Gender"].replace({"male" : 0}, regex=True)
    data["Survived"] = data["Survived"].replace({"Yes" : 1}, regex=True)
    data["Survived"] = data["Survived"].replace({"No" : 0}, regex=True)
    data["NumParentChild"] = data["NumParentChild"].astype(int)
    data["NumSiblingSpouse"] = data["NumSiblingSpouse"].astype(int)
    data["Age"] = data["Age"].astype(float)

    # Using apply() start
    # Pros: Apply custom functions to every elements in the selected column (eg, Passenger Fare, Ticket Class, etc.)
    # Cons: Method of tracking passenger ID
    #       Custom functions does not accept >1 parameters

    # passengerIndex = 0 # Primitive method of tracking Passenger ID, to be changed
    # def roundFare(fare):
    #     global passengerIndex 
    #     passengerIndex = passengerIndex + 1
    #     fare = re.sub("[^0-9.$€£¥₣₹₽₾₺₼₸₴₷฿원₫₮₯₱₳₵₲₪₰]","",fare)
    #     for i in "€£¥₣₹₽₾₺₼₸₴₷฿원₫₮₯₱₳₵₲₪₰":
    #         if i in fare:
    #             fare = input("Other currency symbol detected, key in replacement value (in $) for Passenger ID " + str(passengerIndex) + ": ")
    #     if fare[0] == '$':
    #         fare = fare.replace('$','')
    #     # if fare[0] != '$':
    #     #     response = input("$ header missing for Passenger ID " + str(passengerIndex) + ", key in replacement value? Y/N: ")
    #     #     if response.upper == 'N':
    #     #         fare = input("Please key in replacement value for Passenger ID " + str(passengerIndex) + " (in $): ")
    #     #     if response.upper == 'Y':
    #     #         fare = fare.replace('$','')
    #     #     else:
    #     #         print("Invalid input, keeping values for Passenger ID " + str(passengerIndex))
    #     try:
    #         fare = float(fare)
    #     except ValueError:
    #         fare = -1.00
    #         fare = input("Invalid values detected, key in replacement value (in $) for Passenger ID " + str(passengerIndex) + ": ")
    #     return float(math.floor(fare * 100)/100)

    # def roundAge(age):
    #     return int(age)

    # data["Passenger Fare"] = data["Passenger Fare"].apply(roundFare)
    # # data["Passenger ID"] = data.apply(rowIndex, axis = 1)
    # data["Age"] = data["Age"].apply(roundAge)

    # Using apply() end

    # Without using apply() start
    # Pros: Uses loc member to access specific elements within column, if condition can be used on the element meeting the condition
    # Cons: Seems to take longer for every iteration implemented

    unknownVal = -1

    for x in data.index:
        # Clean Passenger Fare
        data.loc[x, "Passenger Fare"] = re.sub("[^0-9.$]","",data.loc[x, "Passenger Fare"])
        if '$' not in str(data.loc[x, "Passenger Fare"]):
            while True:
                response = input("Missing $ sign for Passenger ID "+str(x+1)+", keep the value? Y/N: ")
                if 'N' in response.upper():
                    try:
                        data.loc[x, "Passenger Fare"] = float(input("Input Passenger Fare for Passenger ID "+str(x+1)+" (in $): "))
                    except ValueError:
                        print("Invalid input for Passenger Fare, setting value to unknownVal")
                        data.loc[x, "Passenger Fare"] = unknownVal
                    break
                if 'Y' in response.upper():
                    break
        elif '$' in str(data.loc[x, "Passenger Fare"]):
            data.loc[x, "Passenger Fare"] = data.loc[x, "Passenger Fare"].replace("$","")
        data.loc[x, "Passenger Fare"] = math.floor(float(data.loc[x, "Passenger Fare"]) * 100)/100
        data.loc[x, "Passenger Fare"] = "{:.2f}".format(data.loc[x, "Passenger Fare"]) # Process takes awhile

        # Clean Ticket Class
        try:
            int(data.loc[x, "Ticket Class"])
        except ValueError:
            data.loc[x, "Ticket Class"] = unknownVal
        if int(data.loc[x, "Ticket Class"]) < 1 or int(data.loc[x, "Ticket Class"]) > 3:
            data.loc[x, "Ticket Class"] = unknownVal

        # Clean Age
        try:
            data.loc[x, "Age"] = math.floor(float(data.loc[x, "Age"]))
        except ValueError:
            data.loc[x, "Age"] = unknownVal
        data.loc[x, "Age"] = int("{:.0f}".format(data.loc[x, "Age"]))
        # data.loc[x, "Age"] = re.sub("[^0-9.]","",str(data.loc[x, "Age"]))
        # data.loc[x, "Age"] = int(math.floor(float(data.loc[x, "Age"])))

        # No clean Ticket Number

        # No clean Cabin

        # Clean Embarkation Country
        if data.loc[x, "Embarkation Country"].__len__() > 1:
            data.loc[x, "Embarkation Country"] = input("Invalid Embarkation Country for Passenger ID "+str(x+1)+", key in correct value (single alphabetical character): ")
        data.loc[x, "Embarkation Country"] = ord(data.loc[x, "Embarkation Country"])

    data['Passenger Fare'] = data['Passenger Fare'].astype(float)
    data['Age'] = data['Age'].astype(int)
    # Without using apply() end
    return data