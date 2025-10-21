from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]          # project root
INP  = ROOT / "data" / "cleaned" / "crashes_clean.csv"  #input output file paths 
OUT  = ROOT / "outcomes" / "descriptives"  
FIGS = ROOT / "outcomes" / "figs"

OUT.mkdir(parents = True, exist_ok = True)
FIGS.mkdir(parents = True, exist_ok = True)



MIN_CRASHES = 500 #minimum crashes per brand 

def make_brand_norm(brand):
    brand_norm = brand.fillna("").astype(str).str.upper().str.strip() 
    brand_norm = brand_norm.str.partition("-")[0].str.strip()   #cuts at hyphen

    brands = {   
        "TOYT":"TOYOTA","NISS":"NISSAN","HOND":"HONDA","HYUN":"HYUNDAI",
        "CHEV":"CHEVROLET","LEXS":"LEXUS","INFI":"INFINITI","VOLK":"VOLKSWAGEN",
        "DODG":"DODGE","MAZD":"MAZDA","SUBA":"SUBARU","CADI":"CADILLAC",
        "LINC":"LINCOLN","BUIC":"BUICK","MERZ":"MERCEDES"
    }
    brand_norm = brand_norm.replace(brands)
    return brand_norm

try: 
    print(f"Reading: {INP}")
    crashesdf = pd.read_csv(INP, low_memory=False)
except:
    print("Error reading file")
    raise SystemExit(1)
#Normalize and starting insights

crashesdf["MAKE_NORM"] = make_brand_norm(crashesdf["VEHICLE_MAKE"]) #normalize brands within df

n_rows = len(crashesdf)
avg_fatal = crashesdf["fatal"].mean() #avg fatal or not
avg_injured = crashesdf["any_injury"].mean() #avg injured or not
# save these in a text file

with open(OUT / "insights.txt", "w") as insighttxt:
    insighttxt.write(f"Number of rows: {n_rows}\n")
    insighttxt.write(f"Average Fatality rate: {avg_fatal}\n")
    insighttxt.write(f"Average Injury rate: {avg_injured}\n")


#injury rate table 


dfRateTab = crashesdf[["MAKE_NORM", "any_injury"]].copy()
dfRateTab = dfRateTab[
    (dfRateTab["MAKE_NORM"] != "") &
    (dfRateTab["MAKE_NORM"] != "UNKNOWN") &
    (dfRateTab["MAKE_NORM"] != "UNK")
]         #no blanks and unknowns

brandTab = (dfRateTab.groupby("MAKE_NORM", as_index = False).agg(crashes = ("any_injury", "size"), injury_rate = ("any_injury", "mean")))

# keep only brands with enough rows, sort descending
brandTab = brandTab[brandTab["crashes"] >= MIN_CRASHES]
brandTab = brandTab.sort_values("crashes", ascending = False)

brandTab.to_csv(OUT / "brandinjurytable.csv", index = False)

#Bar charts for Weekdays/Hourly on crashes 

#Crashes by hour 
hrs = crashesdf["hour"]
wks = crashesdf["weekday"]
hour_counts = [0] * 24          # 0..23
weekday_counts = [0] * 7        # 0..6 

#fill hours index
for v in hrs:
    if 0 <= v <= 23:
        hour_counts[v] += 1

#fill weekdays index
for v in wks:
    if 0 <= v <= 6:
        weekday_counts[v] += 1

#hours
plt.bar(range(24), hour_counts)
plt.xticks(range(24))
plt.title("Crash counts by hour")
plt.xlabel("Hour")
plt.ylabel("Crash count")
plt.savefig(FIGS / "counts_by_hour.png", dpi = 200)
plt.close()

#weekdays
labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
plt.bar(range(7), weekday_counts)
plt.xticks(range(7), labels)
plt.title("Crash counts by weekday")
plt.xlabel("Weekday")
plt.ylabel("Crash count")
plt.savefig(FIGS / "counts_by_weekday.png", dpi = 200)
plt.close()

#filter for crosstabb
keep = crashesdf["VEHICLE_TYPE"].value_counts()
keep = keep[keep >= 50].index      #50 incidents with name probably enough, 150 AND 500 took out alot
tabdf  = crashesdf[crashesdf["VEHICLE_TYPE"].isin(keep)]

#crosstab to show injury rates by vehicleclass
classrates = pd.crosstab(tabdf["VEHICLE_TYPE"], tabdf["any_injury"], normalize = "index")
classrates.columns = ["no_injury_rate", "injury_rate"]
classrates.to_csv(OUT / "injuryratescrosstab.csv", index = True)




