import os
import re
import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------


def _to_utc(ts):
    """

    :param ts:
    :return:
    """
    return pd.to_datetime(ts, utc=True, errors="coerce")


def _days_between(later_utc: pd.Timestamp, earlier_utc: pd.Series) -> pd.Series:
    d = (later_utc - earlier_utc).dt.total_seconds() / (3600 * 24)
    return d.clip(lower=0)


def _bucketize_web(web: pd.DataFrame) -> pd.DataFrame:
    """
    Simple keyword buckets from url+title. First match wins.
    Returns web with extra columns: text, bucket, page_id.
    """
    BUCKETS = {
        "nutrition": [
            "nutrition", "diet", "meal", "recipe", "calorie", "protein",
            "carb", "carbs", "sugar", "glucose", "fiber", "weight", "bmi"
        ],
        "activity": [
            "activity", "exercise", "workout", "fitness", "walk", "walking",
            "steps", "cardio", "strength", "training"
        ],
        "sleep": ["sleep", "insomnia", "circadian", "nap", "bedtime"],
        "stress": ["stress", "anxiety", "mindfulness", "meditation", "breathing", "burnout", "mental"],
        "care": ["doctor", "clinic", "hospital", "appointment", "medication", "pharmacy"],
    }

    web = web.copy()
    web["url"] = web["url"].astype(str)
    web["title"] = web["title"].astype(str)
    web["text"] = (web["url"] + " " + web["title"]).str.lower()

    def assign_bucket(txt: str) -> str:
        for b, kws in BUCKETS.items():
            if any(kw in txt for kw in kws):
                return b
        return "other"

    web["bucket"] = web["text"].apply(assign_bucket)

    # page_id for "unique pages": url if meaningful else title
    web["page_id"] = web["url"]
    bad = web["page_id"].isin(["", "nan", "none", "None"])
    web.loc[bad, "page_id"] = web.loc[bad, "title"]
    return web

def read_and_preprocess(dataset: str, data_root: str, measurement_end: str) -> \
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:

    path = os.path.join(data_root, dataset)
    prefix = "test_" if dataset == "test" else ""

    app = pd.read_csv(os.path.join(path, f"{prefix}app_usage.csv"))
    clm = pd.read_csv(os.path.join(path, f"{prefix}claims.csv"))
    web = pd.read_csv(os.path.join(path, f"{prefix}web_visits.csv"))
    users_path = "churn_labels.csv" if dataset == "train" else "test_members.csv"
    users = pd.read_csv(os.path.join(path, users_path))

    # --- timestamps ---
    app["timestamp"] = pd.to_datetime(app.get("timestamp"), utc=True, errors="coerce")
    web["timestamp"] = pd.to_datetime(web.get("timestamp"), utc=True, errors="coerce")
    clm["diagnosis_date"] = pd.to_datetime(clm.get("diagnosis_date"), errors="coerce")
    users["signup_date"] = pd.to_datetime(users.get("signup_date"), errors="coerce")
    obs_end = pd.to_datetime(measurement_end, utc=True, errors="raise")

    return app, clm, web, users, obs_end

def get_app_features(app: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    """
    App usage features: total sessions, active days, recency. Keep NaNs for missing data.

    """
    app = app.dropna(subset=["member_id"])
    app["date"] = app["timestamp"].dt.floor("D")

    app_agg = app.groupby("member_id").agg(
        app_sessions=("timestamp", "size"),
        app_active_days=("date", "nunique"),
        app_last_ts=("timestamp", "max"),
    ).reset_index()

    app_agg["app_recency_days"] = _days_between(obs_end, app_agg["app_last_ts"])
    app_agg = app_agg.drop(columns=["app_last_ts"])
    return app_agg

def get_web_features(web: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    """
    Web features: overall visits, unique urls, recency + bucketized content features. Keep NaNs for missing data.
    """

    # Overall web features
    web = web.dropna(subset=["member_id"])
    web_overall = web.groupby("member_id").agg(
        web_visits=("timestamp", "size"),
        web_unique_urls=("url", "nunique"),
        web_last_ts=("timestamp", "max"),
    ).reset_index()

    web_overall["web_recency_days"] = _days_between(obs_end, web_overall["web_last_ts"])
    web_overall = web_overall.drop(columns=["web_last_ts"])

    # -------------------------
    # Web content bucket features
    # -------------------------
    web_b = _bucketize_web(web)

    web_bucket = web_b.groupby(["member_id", "bucket"]).agg(
        bucket_visits=("timestamp", "size"),
        bucket_unique=("page_id", "nunique"),
        bucket_last=("timestamp", "max"),
    ).reset_index()

    web_bucket["bucket_recency_days"] = _days_between(obs_end, web_bucket["bucket_last"])
    web_bucket = web_bucket.drop(columns=["bucket_last"])

    # pivot to wide
    visits_w = web_bucket.pivot(index="member_id", columns="bucket", values="bucket_visits")
    uniq_w = web_bucket.pivot(index="member_id", columns="bucket", values="bucket_unique")
    rec_w = web_bucket.pivot(index="member_id", columns="bucket", values="bucket_recency_days")

    visits_w.columns = [f"web_visits_{c}" for c in visits_w.columns]
    uniq_w.columns = [f"web_unique_{c}" for c in uniq_w.columns]
    rec_w.columns = [f"web_recency_{c}_days" for c in rec_w.columns]

    web_content = (
        visits_w.join(uniq_w, how="outer")
        .join(rec_w, how="outer")
        .reset_index()
    )

    # simple "dominant interest" bucket based on visits (keep NaN if no visits)
    dom = visits_w.idxmax(axis=1)
    web_content = web_content.merge(dom.rename("web_bucket_top").reset_index(), on="member_id", how="left")
    web_content["web_bucket_top"] = web_content["web_bucket_top"].astype("category")

    return web_overall, web_content

def get_claim_features(clm: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    # -------------------------
    # Claims features
    # -------------------------
    clm = clm.dropna(subset=["member_id"])
    clm["icd_code"] = clm.get("icd_code", "").astype(str).str.upper().str.strip()

    clm["is_e119"] = clm["icd_code"].str.startswith("E11.9")
    clm["is_i10"] = clm["icd_code"].str.startswith("I10")
    clm["is_z713"] = clm["icd_code"].str.startswith("Z71.3")

    claim_agg = clm.groupby("member_id").agg(
        claims_count=("icd_code", "size"),
        e119=("is_e119", "sum"),
        i10=("is_i10", "sum"),
        z713=("is_z713", "sum"),
        last_dx=("diagnosis_date", "max"),
    ).reset_index()

    # Convert last_dx to UTC for recency (safe even if already datetime)
    last_dx_utc = pd.to_datetime(claim_agg["last_dx"], errors="coerce")
    # localize naive to UTC midnight (best-effort)
    if getattr(last_dx_utc.dt, "tz", None) is None:
        last_dx_utc = last_dx_utc.dt.tz_localize("UTC")
    claim_agg["claims_recency_days"] = _days_between(obs_end, last_dx_utc)
    claim_agg = claim_agg.drop(columns=["last_dx"])

    # binary flags (help sparse counts)
    claim_agg["has_claim"] = (claim_agg["claims_count"] > 0).astype(float)
    claim_agg["has_e119"] = (claim_agg["e119"] > 0).astype(float)
    claim_agg["has_i10"] = (claim_agg["i10"] > 0).astype(float)
    claim_agg["has_z713"] = (claim_agg["z713"] > 0).astype(float)

    return claim_agg


def build_features(
    dataset: str = "train",
    measurement_end: str = "2025-07-14",
    data_root: str = "data",
    feature_set: str = "base"
) -> pd.DataFrame:
    """
    @:param dataset: "train" or "test"
    @:param measurement_end: cutoff date for recency calculations (ISO format)
    @:param data_root: root directory containing "train" and "test" subdirectories with CSV files

    Return: a DataFrame with features for each member_id

    """

    assert dataset in ["train", "test"], "dataset must be 'train' or 'test'"
    assert feature_set in ["base", "base_content", "base_content_intersection"],\
        "feature_set must be 'base', 'base_content', or 'base_content_intersection'"

    app, clm, web, users, obs_end = read_and_preprocess(dataset, data_root, measurement_end)

    app_agg = get_app_features(app, obs_end)
    web_overall, web_content = get_web_features(web, obs_end)
    claim_agg = get_claim_features(clm, obs_end)

    # -------------------------
    # Start from users + merge
    # -------------------------
    if dataset == "train":
        out = users[["member_id", "signup_date", "churn", "outreach"]].copy()
    else:
        out = users[["member_id", "signup_date"]].copy()

    out = out.merge(app_agg, on="member_id", how="left")
    out = out.merge(web_overall, on="member_id", how="left")
    out = out.merge(claim_agg, on="member_id", how="left")
    if "content" in feature_set:
        out = out.merge(web_content, on="member_id", how="left")

    # -------------------------
    # Derived features
    # -------------------------

    # Tenure (days)
    out["tenure_days"] = (obs_end.tz_convert(None) - out["signup_date"]).dt.days
    # Engagement intensity (normalized by tenure)
    out["sessions_per_day"] = out["app_sessions"] / (out["tenure_days"] + 1)
    out["web_visits_per_day"] = out["web_visits"] / (out["tenure_days"] + 1)
    # Web breadth vs depth
    out["web_visits_per_unique_url"] = out["web_visits"] / (out["web_unique_urls"] + 1)
    # App-vs-web mix (0..1); NaN if both missing
    out["app_share_activity"] = out["app_sessions"] / (out["app_sessions"] + out["web_visits"])

    #  interactions: condition × content exposure (very explainable)
    if "intersection" in feature_set:
        out["e119_x_nutrition"] = out["has_e119"] * out.get("web_visits_nutrition")
        out["i10_x_activity"] = out["has_i10"] * out.get("web_visits_activity")
        out["z713_x_nutrition"] = out["has_z713"] * out.get("web_visits_nutrition")

    return out