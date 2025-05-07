# feutils.py  — helper module for the stroke pipeline
def engineer_feats(df):
    df = df.copy()
    df["age_sq"]      = df["age"]**2
    df["glucose_sq"]  = df["avg_glucose_level"]**2
    df["age_glucose"] = df["age"] * df["avg_glucose_level"]
    return df
