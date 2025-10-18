import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the cleaned dataset.
data = pd.read_csv("Motor_Vehicle_Collisions_Cleaned.csv")

# Create the indicator for Ford vehicles.
data['is_ford'] = data['VEHICLE_MAKE'].astype(str).str.lower().str.contains("ford").astype(int)
# Create binary outcome: fatal crash if NUMBER OF PERSONS KILLED > 0.
data['fatal'] = (data['NUMBER OF PERSONS KILLED'] > 0).astype(int)

# --- Estimating Propensity Scores ---
# Define covariates for propensity score estimation.
# Here we include VEHICLE_TYPE and VEHICLE_YEAR. You could add more covariates if they are available.
# For categorical variables, create dummy variables.
covariates = ['VEHICLE_TYPE', 'VEHICLE_YEAR']
ps_data = data[covariates].copy()
ps_data = pd.get_dummies(ps_data, drop_first=True)  # Encode categorical variables.

# Fit logistic regression to estimate propensity scores.
ps_model = LogisticRegression(solver='liblinear')
ps_model.fit(ps_data, data['is_ford'])
data['propensity_score'] = ps_model.predict_proba(ps_data)[:, 1]

# --- Matching using Nearest Neighbors ---
# Define the matching algorithm. We are matching Ford cases (treated) with non‑Ford cases (controls).
# Set n_neighbors = 1 for one-to-one matching.
treated = data[data['is_ford'] == 1]
control = data[data['is_ford'] == 0]

# Reshape the propensity scores for matching.
propensity_treated = treated['propensity_score'].values.reshape(-1, 1)
propensity_control = control['propensity_score'].values.reshape(-1, 1)

# Use nearest neighbor to match each treated unit with a control.
nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(propensity_control)
distances, indices = nn.kneighbors(propensity_treated)

# Get the index of matched controls.
matched_control_indices = control.iloc[indices.flatten()].index
matched_treated_indices = treated.index

# Create a matched dataset.
matched_data = pd.concat([data.loc[matched_treated_indices], data.loc[matched_control_indices]], axis=0)

print("Matched dataset size:", matched_data.shape)
print("Balance check (mean propensity scores):")
print(matched_data.groupby('is_ford')['propensity_score'].mean())

# Re-run logistic regression on the matched data.
X_matched = matched_data[['is_ford']]
X_matched = sm.add_constant(X_matched)
y_matched = matched_data['fatal']
logit_matched = sm.Logit(y_matched, X_matched)
result_matched = logit_matched.fit(disp=False)
print("\nLogistic Regression Results on Matched Data:")
print(result_matched.summary())

# --- Visualization of Propensity Score Distribution ---
plt.figure(figsize=(8, 5))
plt.hist(treated['propensity_score'], bins=25, alpha=0.5, label="Ford Collisions")
plt.hist(control['propensity_score'], bins=25, alpha=0.5, label="Non-Ford Collisions")
plt.xlabel("Propensity Score")
plt.ylabel("Frequency")
plt.title("Distribution of Propensity Scores")
plt.legend()
plt.show()