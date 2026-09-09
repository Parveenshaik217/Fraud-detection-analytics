import numpy as np
import pandas as pd

def psi(reference, current, bins=10):
    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)
    if reference.empty or current.empty:
        return 0.0
    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    r = pd.cut(reference, bins=cuts, include_lowest=True)
    c = pd.cut(current, bins=cuts, include_lowest=True)
    rp = r.value_counts(normalize=True, sort=False).values + 1e-6
    cp = c.value_counts(normalize=True, sort=False).values + 1e-6
    return float(np.sum((cp - rp) * np.log(cp / rp)))

def drift_table(reference, current):
    rows = []
    for col in reference.columns:
        if col in current.columns:
            value = psi(reference[col], current[col])
            status = "OK" if value < .10 else "WATCH" if value < .25 else "DRIFT"
            rows.append({"feature": col, "PSI": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False)
