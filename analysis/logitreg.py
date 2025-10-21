from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# paths
ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "cleaned" / "crashes_clean.csv"
OUT  = ROOT / "outcomes" / "logitreg"
OUT.mkdir(parents = True, exist_ok = True)

#make init for this and str upper in future maybe for reuse across files
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
    crashesdf = pd.read_csv(INP, low_memory = False)
except:
    print("Error reading file")
    raise SystemExit(1)


crashesdf["MAKE_NORM"] = make_brand_norm(crashesdf["VEHICLE_MAKE"])
crashesdf["weekend"] = crashesdf["weekday"].isin([5, 6]).astype(int) # 0-6
crashesdf["hour"] = pd.to_numeric(crashesdf["hour"], errors = "coerce")
crashesdf = crashesdf[crashesdf["hour"].between(0, 23)]
crashesdf = crashesdf[
    (crashesdf["MAKE_NORM"] != "") &
    (crashesdf["MAKE_NORM"] != "UNKNOWN") &
    (crashesdf["MAKE_NORM"] != "UNK")
]  
crashesdf = crashesdf.dropna(subset = ["any_injury", "hour", "MAKE_NORM"]).copy()


MIN_CRASHES = 500
counts = crashesdf.groupby("MAKE_NORM").size() #count
keep = counts[counts >= MIN_CRASHES].index        #more than min
keptdata  = crashesdf[crashesdf['MAKE_NORM'].isin(keep)] #keep

#logit regression
logit = smf.logit('any_injury ~ C(MAKE_NORM, Treatment(reference="FORD")) + weekend + hour', data = keptdata).fit(disp = True)

(OUT / "logit_summary.txt").write_text(logit.summary().as_text())


params = logit.params
conf = logit.conf_int()
brand_levels = sorted([b for b in keep if b != "FORD"])

#odds ratios/confidence intervals df
rows = []
for b in brand_levels:  
    key = f'C(MAKE_NORM, Treatment(reference="FORD"))[T.{b}]'
    if key in params.index:
        rows.append({
            "brand": b,
            "Odds Ratio": float(np.exp(params[key])),
            "Lower CI": float(np.exp(conf.loc[key, 0])),
            "Upper CI": float(np.exp(conf.loc[key, 1])),
        })

OR_table = pd.DataFrame(rows)
OR_table.to_csv(OUT / "Confidence_OddsRatios.csv", index=False)
