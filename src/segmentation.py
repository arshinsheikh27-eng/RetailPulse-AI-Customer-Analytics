
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["RecencyDays", "Frequency", "Monetary"]


def segment_customers(
    customers: pd.DataFrame,
    n_clusters: int = 4
) -> pd.DataFrame:
    """
    Segment customers using log-scaled RFM values and K-Means.

    Returns a copy of the input DataFrame with Segment and SegmentName.
    """

    if customers.empty:
        raise ValueError("Customer data is empty.")

    missing = [col for col in FEATURES if col not in customers.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError("n_clusters must be an integer of at least 2.")

    data = customers.copy()

    # Convert RFM features to numeric.
    x = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)

    # Exclude customers without usable RFM values from model fitting.
    valid_mask = x.notna().all(axis=1)
    valid = x.loc[valid_mask].copy()

    if len(valid) < 2:
        raise ValueError("At least two customers with valid RFM values are required.")

    # RFM values should be non-negative for log1p.
    valid = valid.clip(lower=0)
    valid = np.log1p(valid)

    # Scaling prevents one feature from dominating due to its magnitude.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(valid)

    # Avoid requesting more clusters than usable customers.
    k = min(n_clusters, len(valid))

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_ids = model.fit_predict(scaled)

    # Calculate Silhouette Score when the clustering result is valid.
    unique_labels = np.unique(cluster_ids)

    score = None
    if 1 < len(unique_labels) < len(scaled):
        score = float(silhouette_score(scaled, cluster_ids))

    data["Segment"] = pd.Series(
        pd.NA, index=data.index, dtype="object"
    )
    data.loc[valid_mask, "Segment"] = cluster_ids.astype(str)

    # Use original RFM values to interpret each cluster.
    profile = data.loc[valid_mask, FEATURES].copy()
    profile["Segment"] = data.loc[valid_mask, "Segment"]

    summary = profile.groupby("Segment")[FEATURES].mean()

    # Rank clusters using all three RFM dimensions:
    # lower recency is better; higher frequency and monetary are better.
    ranks = pd.DataFrame(index=summary.index)

    ranks["RecencyScore"] = summary["RecencyDays"].rank(
        ascending=False, pct=True
    )
    ranks["FrequencyScore"] = summary["Frequency"].rank(
        ascending=True, pct=True
    )
    ranks["MonetaryScore"] = summary["Monetary"].rank(
        ascending=True, pct=True
    )

    ranks["OverallScore"] = ranks[
        ["RecencyScore", "FrequencyScore", "MonetaryScore"]
    ].mean(axis=1)

    ordered_segments = ranks["OverallScore"].sort_values().index.tolist()

    names = ["Lapsed", "At Risk", "Engaged", "Recent/Active"]

    # Adapt names when the selected cluster count differs from four.
    if len(ordered_segments) == 2:
        names = ["Lower Engagement", "Higher Engagement"]
    elif len(ordered_segments) == 3:
        names = ["Lapsed/At Risk", "Engaged", "Recent/Active"]
    elif len(ordered_segments) > 4:
        names = [
            f"Segment {i + 1}" for i in range(len(ordered_segments))
        ]

    label_map = dict(zip(ordered_segments, names))

    data["SegmentName"] = data["Segment"].map(label_map)

    # Store the Silhouette Score for dashboard display.
    data.attrs["silhouette_score"] = score

    return data
