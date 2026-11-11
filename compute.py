import pandas as pd
import numpy as np
# ------ load + clean data ------
df = pd.read_excel("data_combined.xlsx")
df = df[df["City"].notna()].copy()
df = df.set_index("City")
df = df.apply(pd.to_numeric, errors= "coerce")

direction = {
    "grad_rate": "higher", "student_teacher": "lower",
    "violent_crime": "lower", "ped_fatality": "lower",
    "housing_burden": "lower",
    "park_spend": "higher", "park_access": "higher",
    "transit": "higher", "walk": "higher",
    "primary_care": "higher", "mental_health": "higher",
    "pm25": "lower",
    "disconnected_youth": "lower", "college_access": "higher",
}


# ------ normalize data ------
norm = pd.DataFrame(index = df.index)
for col, pref in direction.items():
    x = df[col]
    lo, hi = x.min(), x.max()
    #skip empty columns and ones where every city has the same value so no division by 0
    if x.isna().all() or hi == lo:
        print(f"skipping {col}")
        continue
    scaled = (x - lo) / (hi - lo)
    norm[col] = scaled if pref == "higher" else 1 - scaled

print(norm.round(2))

# -----entropy weighting-----
n = len(norm)

p = norm / norm.sum()
plogp = (p * np.log(p)).fillna(0) # 0 * log(0) = NaN, so treat it as 0
entropy = -plogp.sum() / np.log(n)
divergence = 1 - entropy
weight = divergence / divergence.sum()

print(weight.sort_values(ascending=False).round(3))
print("sum:", weight.sum())

score = (norm * weight).sum(axis=1).sort_values(ascending=False)
print(score.round(3))