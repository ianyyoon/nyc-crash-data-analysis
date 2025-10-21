# nyc-crash-data-analysis
Exploratory analysis of crash injury risk in NYC by vehicle make vs. injury odds, and time-of-day patterns
Main report available NYC_Crashes_Analysis_Report.pdf

# Environment
- Python 3.10.6
- Install depencies via: pip install -r requirements.txt

How to run:
1.Install dependencies in requirements.txt (above)
2.Run data_cleanup.py, descriptives.py, logitreg.py(data_cleanup first) in analysis/
3.Results will be written out in data/cleaned/ and outcomes in descriptives, figs(bar charts), logitreg folders.
4.My report writeup is available as NYC_Crashes_Analysis_Report in the parent repo folder

Outputs:
data/cleaned/crashes_clean.csv
outcomes/figs/counts_by_hour.png
outcomes/figs/counts_by_weekday.png
outcomes/descriptives/brandinjurytable.csv
outcomes/descriptives/injuryratescrosstab.csv
outcomes/logitreg/logit_summary.txt
outcomes/logitreg/Confidence_OddsRatios.csv

Logit Model notes:
Odds ratios with confidence intervals, and log odds as coefficients
Significant Values: p < 0.05 and CI excludes 1.00

Future improvements to make:
weighing of other available variables like body type, road, etc within logit regression model

adding methods file for repeated cleaning functions, and standardizing them per column in the future

add significance flags to code when making odds table to avoid extra work on report next time



