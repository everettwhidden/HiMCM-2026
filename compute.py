import pandas as pd
import numpy as np

# ------ load + clean data ------
df = pd.read_excel("data_combined.xlsx")
df = df[df["City"].notna()].copy()
df = df.set_index("City")
df = df.apply(pd.to_numeric, errors="coerce")

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

# which indicators belong to which criterion
criteria = {
    "Education": ["grad_rate", "student_teacher"],
    "Safety": ["violent_crime", "ped_fatality"],
    "Affordability": ["housing_burden"],
    "Recreation": ["park_spend", "park_access"],
    "Independence": ["transit", "walk"],
    "Health": ["primary_care", "mental_health"],
    "Environment": ["pm25"],
    "Future Opportunities": ["disconnected_youth", "college_access"],
}

# ------ normalize indicators (r_ij) ------
norm = pd.DataFrame(index=df.index)
for col, pref in direction.items():
    x = df[col]
    lo, hi = x.min(), x.max()
    # skip empty columns and ones where every city has the same value so no division by 0
    if x.isna().all() or hi == lo:
        print(f"skipping {col}")
        continue
    scaled = (x - lo) / (hi - lo)
    norm[col] = scaled if pref == "higher" else 1 - scaled

# ------ criterion scores (C_ik) ------
# average the indicators in each criterion. mean() skips NaN, so a missing
# indicator just drops out of the average instead of counting as 0
C = pd.DataFrame(index=norm.index)
for crit, cols in criteria.items():
    cols = [c for c in cols if c in norm.columns]
    C[crit] = norm[cols].mean(axis=1)

# a criterion with no data at all would get entropy 0 and the biggest weight,
# so drop it instead
empty = C.columns[C.isna().all()]
for crit in empty:
    print(f"no data for {crit}, dropping it")
C = C.drop(columns=empty)
print(C.round(2))

# ------ entropy weights (w_k^E) ------
n = len(C)
p = C / C.sum()
with np.errstate(divide="ignore", invalid="ignore"):
    plogp = (p * np.log(p)).fillna(0)  # 0 * log(0) = NaN, treat it as 0
E = -plogp.sum() / np.log(n)
w_E = (1 - E) / (1 - E).sum()

print(w_E.sort_values(ascending=False).round(3))
print("sum:", w_E.sum())

# ------ final weights (w_k) ------
# entropy only for now. once AHP is done: w = lam * w_A + (1 - lam) * w_E
w = w_E

# ------ SAW score (S_i), 0-100 ------
S = 100 * (C * w).sum(axis=1)
print(S.sort_values(ascending=False).round(1))