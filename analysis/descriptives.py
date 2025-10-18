from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]          # project root
INP  = ROOT / "data" / "cleaned" / "crashes_clean.csv"  #input output file paths 
OUT  = ROOT / "outcomes" / "descriptives"  
#OUT2  = ROOT / "outcomes" / "logitreg"  / "logitreg.csv"  for logit file
OUT.parent.mkdir(parents=True, exist_ok=True)
#OUT2.parent.mkdir(parents=True, exist_ok=True)             for logit file


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

cleandf = pd.read_csv(INP) 

cleandf["MAKE_NORM"] = make_brand_norm(cleandf["VEHICLE_MAKE"])
cleandf["isford"]   = (cleandf["MAKE_NORM"] == "FORD").astype(int)

len(cleandf)
cleandf["fatal"].mean()
# save these in a text file














data['fatal'] = (data['NUMBER OF PERSONS KILLED'] > 0).astype(int)

crosstab = pd.crosstab(data['is_ford'], data['fatal'], margins=True)
print("Contingency Table: Ford Involvement vs Fatal Crash Outcome")
print(crosstab)

X = data[['is_ford']]
X = sm.add_constant(X)
y = data['fatal']

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=True)

print("\nLogistic Regression Results:")
print(result.summary())

odds_ratio = np.exp(result.params['is_ford'])
print("\nOdds Ratio for Ford Vehicle Involvement:", odds_ratio)

data['predicted_prob'] = result.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(data['is_ford'], data['predicted_prob'], alpha=0.3, label="Predicted Probability")
plt.xlabel('Ford Vehicle (0 = No, 1 = Yes)')
plt.ylabel('Predicted Probability of Fatal Crash')
plt.title('Predicted Fatal Crash Probability by Ford Vehicle Involvement')
plt.legend()
plt.grid(True)
plt.show()

fatality_rate = data.groupby('is_ford')['fatal'].mean()
print("\nFatality Rates by Ford Vehicle Involvement:")
print(fatality_rate)