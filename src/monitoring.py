import pandas as pd
import numpy as np

def numeric_drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Lightweight PSI drift report. Current/reference data should represent comparable windows."""
    columns = columns or [c for c in reference.select_dtypes(include="number").columns if c in current.columns]
    rows = []
    for col in columns:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(ref) < 20 or len(cur) < 20:
            continue
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        rp = np.histogram(ref, bins=edges)[0] / len(ref)
        cp = np.histogram(cur, bins=edges)[0] / len(cur)
        rp = np.clip(rp, 1e-6, None)
        cp = np.clip(cp, 1e-6, None)
        psi = float(np.sum((cp-rp) * np.log(cp/rp)))
        rows.append({"Feature": col, "ReferenceMean": ref.mean(), "CurrentMean": cur.mean(),
                     "PSI": psi, "Status": "Potential drift" if psi >= .2 else "No strong drift signal"})
    return pd.DataFrame(rows)
