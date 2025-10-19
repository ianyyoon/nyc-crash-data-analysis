from pathlib import Path #to avoid hardcoding paths like before
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # project root
INP  = ROOT / "data" / "original" / "Motor_Vehicle_Collisions_-_Crashes.csv"  #input output file paths 
OUT  = ROOT / "data" / "cleaned"  / "crashes_clean.csv"
OUT.parent.mkdir(parents = True, exist_ok = True)

# load dataset, exception if can't read file

try: 
    print(f"Reading: {INP}")
    data = pd.read_csv(INP, low_memory = False)
except:
    print("Error reading file")
    raise SystemExit(1)


#column header list:  [STATE_REGISTRATION	VEHICLE_TYPE	VEHICLE_MAKE	VEHICLE_MODEL	VEHICLE_YEAR	TRAVEL_DIRECTION	VEHICLE_OCCUPANTS	DRIVER_SEX	DRIVER_LICENSE_STATUS	DRIVER_LICENSE_JURISDICTION	PRE_CRASH	POINT_OF_IMPACT	VEHICLE_DAMAGE	VEHICLE_DAMAGE_1	VEHICLE_DAMAGE_2	VEHICLE_DAMAGE_3	PUBLIC_PROPERTY_DAMAGE	PUBLIC_PROPERTY_DAMAGE_TYPE	CONTRIBUTING_FACTOR_1	CONTRIBUTING_FACTOR_2	NUMBER OF PERSONS INJURED	NUMBER OF PERSONS KILLED]
#could use classes/ functions for alot of the repeated processes below in the future


data = data.dropna(subset = ['COLLISION_ID']) #drop any blank collision ids, all of them looked fine

if 'VEHICLE_ID' in data.columns:
    data = data.drop_duplicates(subset = ['COLLISION_ID', 'VEHICLE_ID']) #ensure one row per vehicle 

#CRASH_DATE to datetime + drop empty CRASH_DATE + seperate weekday column 
if 'CRASH_DATE' in data.columns:
    data['CRASH_DATE'] = pd.to_datetime(data['CRASH_DATE'], errors = 'coerce')
    data = data.dropna(subset = ['CRASH_DATE'])
    data['weekday'] = data['CRASH_DATE'].dt.dayofweek

#CRASH_TIME to hour column, less strict then date   
if 'CRASH_TIME' in data.columns:
    data['hour'] = pd.to_numeric(data['CRASH_TIME'].str.split(':').str[0], errors = 'coerce')

# Standardize VEHICLE_MAKE 
if 'VEHICLE_MAKE' in data.columns:
    data['VEHICLE_MAKE'] = data['VEHICLE_MAKE'].astype(str).str.strip().str.upper()


# Standardize VEHICLE_Model 
if 'VEHICLE_MODEL' in data.columns:
    data['VEHICLE_MODEL'] = data['VEHICLE_MODEL'].astype(str).str.strip().str.upper()

# drop any blank fatalities and add boolean to filter between which instances had or didnt have fatalities
if 'NUMBER OF PERSONS KILLED' in data.columns:
    data = data.dropna(subset = ['NUMBER OF PERSONS KILLED']) #drop blanks, werent any but just in case
    data['fatal'] = (data['NUMBER OF PERSONS KILLED'] > 0).astype('int') #bool return on 1 or more fatal for easy filtering then cast to int so no conversion needed for avging
else:
    print("Missing 'NUMBER OF PERSONS KILLED'") #pretty sure the dataset didnt have any fatalities

if 'NUMBER OF PERSONS INJURED' in data.columns:
    data = data.dropna(subset = ['NUMBER OF PERSONS INJURED'])  
    data['any_injury'] = (data['NUMBER OF PERSONS INJURED'] > 0).astype(int)  
else:
    print("Missing 'NUMBER OF PERSONS INJURED'") #this one there was a handful


# Save the cleaned dataset to a new CSV file
data.to_csv(OUT, index = False)
print("Cleaned data saved as 'crashes_clean.csv'.")

