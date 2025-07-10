# === PART A: Imports + Function Setup app/model_logic.py ===

import pandas as pd
import numpy as np
import os
import warnings
import logging
warnings.filterwarnings("ignore")

# ⛔️ DO NOT import heavy packages here ⛔️
# Move CatBoost, SHAP, BERT, UMAP, and KMeans into the function body

def run_full_pipeline(file_path: str) -> pd.DataFrame:
    logging.debug("Starting run_full_pipeline")
    # ✅ Lazy load heavy packages here
    import shap
    from catboost import CatBoostClassifier
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from sklearn.cluster import KMeans
    import requests
    from io import BytesIO

    # === PART B: Load Model + Data + Initial Cleanup ===

    try:
        print("📦 Loading CatBoost model...")
        model = CatBoostClassifier()
        model.load_model("models/catboost_v2_model.cbm")
        print("✅ CatBoost model loaded")
    except Exception as e:
        print("❌ Failed to load CatBoost model:", str(e))
        raise

    try:
        print("📦 Downloading SentenceTransformer...")
        model_bert = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ SentenceTransformer loaded")
    except Exception as e:
        print("❌ Failed to load BERT model:", str(e))
        raise

    test_df = pd.read_csv(file_path, encoding='ISO-8859-1')
    test_df.columns = test_df.columns.str.strip()
    original_columns = test_df.columns.tolist()

    # Clean numeric columns
    comma_cols = ["Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM", "Net Amount"]
    for col in comma_cols:
        if col in test_df.columns:
            test_df[col] = test_df[col].astype(str).str.replace(",", "").replace("nan", np.nan).astype(float)

    # Combine text fields
    text_fields = ["Line Desc", "Source Desc", "Batch Name"]
    test_df[text_fields] = test_df[text_fields].fillna("")
    test_df["Combined_Text"] = test_df["Line Desc"] + " | " + test_df["Source Desc"] + " | " + test_df["Batch Name"]

    # === PART C: BERT Embeddings + UMAP + Clustering ===
    embeddings = model_bert.encode(test_df["Combined_Text"].tolist(), show_progress_bar=False)
    embedding_df = pd.DataFrame(embeddings, columns=[f"text_emb_{i}" for i in range(embeddings.shape[1])])
    test_df = pd.concat([test_df.reset_index(drop=True), embedding_df], axis=1)

    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    reduced = umap_model.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=10, random_state=42)
    test_df["Narration_Cluster"] = kmeans.fit_predict(reduced)

    cluster_summary = (
        test_df.groupby("Narration_Cluster")["Combined_Text"]
        .apply(lambda x: "; ".join(x.head(3)))
        .reset_index(name="Narration_Cluster_Label")
    )
    test_df = test_df.merge(cluster_summary, on="Narration_Cluster", how="left")




# === PART D: Date Features + Feature Preparation ===
    date_cols = ["Accounting Date", "Invoice Date", "Posted Date"]
    for col in date_cols:
        test_df[col] = pd.to_datetime(test_df[col], errors="coerce")

    test_df["Accounting_Month"] = test_df["Accounting Date"].dt.month
    test_df["Accounting_Weekday"] = test_df["Accounting Date"].dt.weekday
    test_df["Invoice_Month"] = test_df["Invoice Date"].dt.month
    test_df["Invoice_Weekday"] = test_df["Invoice Date"].dt.weekday
    test_df["Posted_Month"] = test_df["Posted Date"].dt.month
    test_df["Posted_Weekday"] = test_df["Posted Date"].dt.weekday

    exclude_cols = ["S. No", "Combined_Text", "Accounting Date", "Invoice Date", "Posted Date"]
    model_feature_names = model.feature_names_
    feature_cols = [col for col in test_df.columns if col in model_feature_names and col not in exclude_cols and not col.startswith("Unnamed")]

    for col in feature_cols:
        if test_df[col].dtype == object or test_df[col].isnull().any():
            test_df[col] = test_df[col].astype(str).fillna("Missing")

    X_final = test_df[feature_cols].copy()

#=== PART E: CatBoost Predictions + SHAP + Risk Summaries
    logging.debug("Before model prediction")
    # === Model Predictions ===
    test_df["Model_Score"] = model.predict_proba(X_final)[:, 1]
    test_df["Final_Score"] = test_df["Model_Score"].round(3)
    logging.debug("After model prediction")

    logging.debug(">>> Starting SHAP/explainability section")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_final)

    # === Risk Explanation Columns ===
    amount_features = ["Entered Dr SUM", "Entered Cr SUM", "Accounted Dr SUM", "Accounted Cr SUM", "Net Amount"]
    date_features = ["Accounting_Month", "Accounting_Weekday", "Invoice_Month", "Invoice_Weekday", "Posted_Month", "Posted_Weekday"]
    account_info_features = ["Account Name", "Nature in balance sheet", "Source Name", "Document Type", "Tax Rate", "Tax Rate Name"]
    other_features = [col for col in model_feature_names if col not in amount_features + date_features + account_info_features and not col.startswith("text_emb_")]

    feature_groups = {
        "Amount": amount_features,
        "Date": date_features,
        "Source Info": account_info_features,
        "Batch": other_features,
        "Narration": ["Narration_Cluster_Label"]
    }

    explanation_templates = {
        "Narration": "Narration pattern resembles high-value or structured payouts",
        "Amount": "High {feature} = ₹{value:,.0f}",
        "Date": "Posted on {feature} = {value}",
        "Source Info": "{feature} = '{value}' is missing or looks suspicious",
        "Batch": "Batch reference '{value}' appears frequently in vendor payments"
    }

    top_risky_texts, top_safe_texts, final_summaries = [], [], []
    for i in range(len(X_final)):
        row_shap = shap_values[i]
        row = test_df.iloc[i]
        score = row["Final_Score"]
        level = "HIGH RISK" if score >= 0.8 else ("MEDIUM RISK" if score >= 0.5 else "LOW RISK")

        impact_by_group = {}
        feature_info = {}
        for group, features in feature_groups.items():
            valid_feats = [f for f in features if f in feature_cols]
            if not valid_feats:
                continue
            group_shap_sum = sum(row_shap[feature_cols.index(f)] for f in valid_feats)
            impact_by_group[group] = group_shap_sum
            top_feat = max(valid_feats, key=lambda f: abs(row_shap[feature_cols.index(f)]))
            value = row.get(top_feat, "N/A")
            feature_info[group] = (top_feat, value)

        sorted_risk = sorted(impact_by_group.items(), key=lambda x: x[1], reverse=True)
        sorted_safe = sorted(impact_by_group.items(), key=lambda x: x[1])

        def render(group, feature, value):
            if group == "Narration":
                return explanation_templates[group]
            elif group in explanation_templates:
                return explanation_templates[group].format(feature=feature, value=value)
            else:
                return f"{group}: {feature} = {value}"

        top_risk = [render(g, *feature_info[g]) for g, _ in sorted_risk[:3]]
        top_safe = [render(g, *feature_info[g]) for g, _ in sorted_safe if g not in [r[0] for r in sorted_risk[:3]][:2]]

        top_risky_texts.append("\n".join(f"- {t}" for t in top_risk))
        top_safe_texts.append("\n".join(f"- {t}" for t in top_safe[:2]))

        enriched = f"⚠️ This transaction is classified as **{level}** with a score of {score:.2f}.\n\nTop contributors to risk:\n" + \
                   "\n".join(f"{t}" for t in top_risk) + \
                   "\n\nRisk reduced by:\n" + "\n".join(f"{t}" for t in top_safe[:2])
        final_summaries.append(enriched)

    test_df["Top_Risky_Feature_Groups"] = top_risky_texts
    test_df["Top_Safe_Feature_Groups"] = top_safe_texts
    test_df["Explanation_Summary"] = final_summaries
    logging.debug(">>> Finished SHAP/explainability section")

#=== PART F: Control Point Logic (CP_01 to CP_32) + Final Return

    # === CONTROL POINTS SETUP ===
    cp_score_dict = {
        "CP_01": 83, "CP_02": 86, "CP_03": 78, "CP_04": 81, "CP_07": 84, "CP_08": 80,
        "CP_09": 76, "CP_15": 88, "CP_16": 73, "CP_17": 75, "CP_19": 60,
        "CP_21": 69, "CP_22": 66, "CP_23": 87, "CP_24": 78, "CP_26": 0,
        "CP_30": 72, "CP_32": 72
    }
    valid_cps = list(cp_score_dict.keys())

    pl_net_total = test_df[test_df["PL/ BS"] == "PL"]["Net Amount"].abs().sum()
    pl_net_threshold = 0.10 * pl_net_total
    total_net = test_df["Net Amount"].abs().sum()

    def cp_01(row):
        keywords = ['fraud','bribe','kickback','suspicious','fake','dummy','gift','prize','token','reward','favour']
        text = f"{str(row.get('Line Desc', '')).lower()} {str(row.get('Source Desc', '')).lower()}"
        return int(any(k in text for k in keywords))

    def cp_02(row):
        return int(row.get("PL/ BS") == "PL" and abs(row.get("Net Amount", 0)) > pl_net_threshold)

    def cp_03_flags(df):
        a = df.duplicated(subset=["Accounting Date", "Line Desc", "Source Desc", "Source Name"], keep=False)
        b = df.duplicated(subset=["Accounting Date", "Account Code", "Net Amount"], keep=False)
        c = df.duplicated(subset=["Document Number"], keep=False) & ~df.duplicated(subset=["Accounting Date", "Document Number"], keep=False)
        d = df.duplicated(subset=["Accounting Date", "Line Desc", "Account Code"], keep=False)
        return ((a | b | c | d).astype(int))

    def cp_04(row): return cp_02(row)

    def cp_07_flags(df): return (df.groupby("Document Number")["Net Amount"].transform("sum").round(2) != 0).astype(int)

    def cp_08(row):
        text = f"{row.get('Account Name', '')} {row.get('Line Desc', '')} {row.get('Source Desc', '')}".lower()
        return int("cash in hand" in text)

    def cp_09_flags(df):
        result = pd.Series(0, index=df.index)
        for doc_id, group in df.groupby("Document Number"):
            accs = group["Account Name"].dropna().str.lower().tolist()
            if any("cash" in a for a in accs) and any("bad debt" in a for a in accs):
                result[group.index] = 1
        return result

    def cp_15_flags(df):
        grp_sum = df.groupby(["Account Code", "Accounting Date"])[["Entered Dr SUM", "Entered Cr SUM"]].sum().sum(axis=1)
        keys = grp_sum[grp_sum > 0.03 * total_net].index
        return df.set_index(["Account Code", "Accounting Date"]).index.isin(keys).astype(int)

    def cp_16_flags(df):
        if "Currency" not in df.columns:
            df["Currency"] = "INR"
        docs = df.groupby("Document Number")["Currency"].nunique()
        flagged = docs[docs > 1].index
        return df["Document Number"].isin(flagged).astype(int)

    def cp_17_flags(df):
        sums = df[df["PL/ BS"] == "PL"].groupby("Source Name")["Net Amount"].sum().abs()
        risky = sums[sums > 0.03 * pl_net_total].index
        return df["Source Name"].isin(risky).astype(int)

    def cp_19(row):
        try: return int(pd.to_datetime(row["Accounting Date"]).weekday() == 6)
        except: return 0

    def cp_21(row):
        try:
            date = pd.to_datetime(row.get("Accounting Date"))
            return int(date == (date + pd.offsets.MonthEnd(0)))
        except: return 0

    def cp_22(row):
        try:
            date = pd.to_datetime(row.get("Accounting Date"))
            return int(date.day == 1)
        except: return 0

    def cp_23(row):
        text = f"{row.get('Line Desc', '')} {row.get('Account Name', '')}".lower()
        return int(any(t in text for t in ['derivative', 'spv', 'structured', 'note', 'swap']))

    def cp_24(row):
        try:
            last = str(int(abs(row.get("Net Amount", 0))))[-3:]
            seqs = {'123','234','345','456','567','678','789','890','321','432','543','654','765','876','987','098'}
            repeats = {str(i)*3 for i in range(10)} | {'000'}
            return int(last in seqs or last in repeats and last != '901')
        except: return 0

    def cp_26_flags(df):
        try:
            doc_ids = sorted(df["Document Number"].dropna().astype(int).unique())
            missing = {doc_ids[i]+1 for i in range(len(doc_ids)-1) if doc_ids[i+1] - doc_ids[i] > 1}
            flagged = set()
            for miss in missing:
                flagged.update([miss-1, miss+1])
            return df["Document Number"].astype(int).isin(flagged).astype(int)
        except: return pd.Series(0, index=df.index)

    def cp_30(row):
        text = f"{row.get('Line Desc', '')} {row.get('Account Name', '')}".lower()
        return int(any(t in text for t in ['derivative','option','swap','future','structured']))

    def cp_32(row): return int(row.get("Net Amount", 0) == 0)

#=== PART G (Final): Apply CPs + Score + Return

    test_df["CP_01"] = test_df.apply(cp_01, axis=1)
    test_df["CP_02"] = test_df.apply(cp_02, axis=1)
    test_df["CP_03"] = cp_03_flags(test_df)
    test_df["CP_04"] = test_df.apply(cp_04, axis=1)
    test_df["CP_07"] = cp_07_flags(test_df)
    test_df["CP_08"] = test_df.apply(cp_08, axis=1)
    test_df["CP_09"] = cp_09_flags(test_df)
    test_df["CP_15"] = cp_15_flags(test_df)
    test_df["CP_16"] = cp_16_flags(test_df)
    test_df["CP_17"] = cp_17_flags(test_df)
    test_df["CP_19"] = test_df.apply(cp_19, axis=1)
    test_df["CP_21"] = test_df.apply(cp_21, axis=1)
    test_df["CP_22"] = test_df.apply(cp_22, axis=1)
    test_df["CP_23"] = test_df.apply(cp_23, axis=1)
    test_df["CP_24"] = test_df.apply(cp_24, axis=1)
    test_df["CP_26"] = cp_26_flags(test_df)
    test_df["CP_30"] = test_df.apply(cp_30, axis=1)
    test_df["CP_32"] = test_df.apply(cp_32, axis=1)

    def compute_cp_score(row):
        triggered = [cp for cp in valid_cps if row.get(cp, 0) == 1]
        if not triggered: return 0.0
        product = 1.0
        for cp in triggered:
            product *= (1 - cp_score_dict[cp] / 100)
        return round(1 - product, 4)

    def list_triggered_cps(row):
        return ", ".join([f"{cp} ({cp_score_dict[cp]})" for cp in valid_cps if row.get(cp, 0) == 1])

    test_df["Triggered_CPs"] = test_df.apply(list_triggered_cps, axis=1)
    test_df["CP_Score"] = test_df.apply(compute_cp_score, axis=1)

    # Drop 384 BERT columns from output
    test_df = test_df.drop(columns=[col for col in test_df.columns if col.startswith("text_emb_")])
    logging.debug("run_full_pipeline complete")
    return test_df




