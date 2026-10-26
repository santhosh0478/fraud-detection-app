import json
import os

import pandas as pd
import streamlit as st

from fraud_core import FraudModels

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Fraud Detection Demo", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_models():
    return FraudModels(HERE)            # loaded once, reused for every request


@st.cache_data
def load_examples():
    gold = pd.read_csv(os.path.join(HERE, "golden_scores.csv"))
    drop = ["xgb_prob", "ae_score", "xgb_flag", "ae_flag", "y_true"]
    return gold[gold["y_true"] == 1].drop(columns=drop), gold[gold["y_true"] == 0].drop(columns=drop)


models = load_models()
fraud_examples, normal_examples = load_examples()
st.session_state.setdefault("tx_json", "")


def put_example(kind):
    pool = fraud_examples if kind == "fraud" else normal_examples
    st.session_state["tx_json"] = json.dumps(pool.sample(1).iloc[0].to_dict())


RISK = {
    "HIGH": (st.error, "HIGH RISK: flagged as likely fraud by XGBoost"),
    "REVIEW": (st.warning, "REVIEW: XGBoost says normal, but the autoencoder finds it unusual (low-confidence signal)"),
    "LOW": (st.success, "LOW RISK: looks genuine"),
}


def show_result(out):
    show, message = RISK[out["risk_level"]]
    show(message)
    c1, c2 = st.columns(2)
    c1.metric("XGBoost fraud probability", f"{out['xgb_probability']:.4f}",
              help=f"Flagged when at or above {out['xgb_threshold']:.4f}")
    if "ae_score" in out:
        c2.metric("Autoencoder anomaly score", f"{out['ae_score']:.3f}",
                  help=f"Flagged when at or above {out['ae_threshold']:.3f}")
    st.caption("Top XGBoost feature contributions (log-odds; positive pushes toward fraud). "
               "They explain what drove the score. They are not proof of fraud.")
    st.bar_chart(pd.DataFrame(out["xgb_top_drivers"]).set_index("feature"))
    if "ae_top_errors" in out:
        st.caption("Features the autoencoder reconstructs worst for this transaction:")
        st.dataframe(pd.DataFrame(out["ae_top_errors"]))


st.title("🛡️ Fraud Detection Demo")
st.write("**XGBoost** (supervised) makes the main decision and a PyTorch-trained **autoencoder** (unsupervised) "
         "gives a second opinion. Trained on the public ULB credit-card dataset (0.17% fraud). "
         "**Demo only: not for real financial decisions. Nothing you enter is stored.**")

with st.sidebar:
    st.header("Model settings")
    st.write(f"XGBoost flag threshold: **{models.xgb_thr:.4f}**")
    st.write(f"Autoencoder flag threshold: **{models.ae_thr:.3f}**")
    st.caption("Thresholds were chosen on a validation set, never on the test set.")

tab1, tab2, tab3, tab4 = st.tabs(["Check one transaction", "Check a CSV file", "Model results", "About and limitations"])

with tab1:
    st.write("Paste one transaction as JSON (fields V1 to V28 and Amount; Time is optional), "
             "or load a held-out test example.")
    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("Load a genuine example", on_click=put_example, args=("normal",))
    b2.button("Load a fraud example", on_click=put_example, args=("fraud",))
    text = st.text_area("Transaction JSON", key="tx_json", height=220)
    if st.button("Check transaction", type="primary"):
        try:
            feats = json.loads(text)
            feats = feats.get("features", feats)
            out = models.predict_one(feats)
        except Exception as e:
            st.error(f"Could not score this input: {e}")
        else:
            show_result(out)

with tab2:
    st.write("Upload a CSV with columns V1 to V28 and Amount (Time and Class are ignored).")
    uploaded = st.file_uploader("CSV file", type="csv")
    use_sample = st.checkbox("Use the included sample file (50 transactions, 15 of them real frauds)")
    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        data = pd.read_csv(os.path.join(HERE, "sample_transactions.csv"))
    if data is not None:
        if len(data) > 5000:
            st.warning("Only the first 5,000 rows are scored in this demo.")
            data = data.head(5000)
        data = data.drop(columns=["Class"], errors="ignore")
        try:
            res = models.predict_df(data)
        except Exception as e:
            st.error(f"Could not score this file: {e}")
        else:
            counts = res["risk_level"].value_counts()
            m1, m2, m3 = st.columns(3)
            m1.metric("HIGH risk", int(counts.get("HIGH", 0)))
            m2.metric("REVIEW", int(counts.get("REVIEW", 0)))
            m3.metric("LOW risk", int(counts.get("LOW", 0)))
            table = res.copy()
            table.insert(0, "row", range(len(table)))
            table["xgb_probability"] = table["xgb_probability"].round(4)
            if "ae_score" in table:
                table["ae_score"] = table["ae_score"].round(3)
            st.dataframe(table)
            st.download_button("Download results (CSV)", table.to_csv(index=False), "fraud_results.csv", "text/csv")

with tab3:
    st.subheader("Held-out test results")
    results_path = os.path.join(HERE, "results.csv")
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        st.dataframe(results)
        for _, r in results[results["split"] == "test"].iterrows():
            st.write(f"**{r['model']}** (test): precision {r['precision']:.2f}, recall {r['recall']:.2f}, "
                     f"PR-AUC {r['pr_auc']:.2f}")
    st.markdown(
        "- **Thresholds** were picked on the validation set; the test set was used only to report results.\n"
        "- **PR-AUC** (area under the precision-recall curve) is the headline metric because accuracy is "
        "misleading when only 0.17% of transactions are fraud.\n"
        "- **XGBoost sees fraud labels while training; the autoencoder and Isolation Forest do not.** "
        "That is why XGBoost scores much higher, and why the unsupervised models are only a benchmark here.")

with tab4:
    st.markdown(
        "**What this is:** a portfolio project, not a production fraud system.\n\n"
        "**Limitations**\n"
        "- The data is public, anonymised (PCA features V1 to V28) and covers two days of European card "
        "transactions in 2013. Real-world performance is unknown.\n"
        "- The split is random, not by time, and the test set has only 95 frauds, so every number is noisy "
        "(for example, 74 of 76 flagged being correct is consistent with a true precision of roughly 91% to 99%).\n"
        "- XGBoost can only recognise patterns similar to past labelled frauds. New fraud tactics may be missed.\n"
        "- The autoencoder's extra REVIEW flag is weak: in testing only about 2 of 30 REVIEW rows were real frauds. "
        "Treat it as a low-priority queue, not an alert.\n"
        "- Feature contributions show what drove a score. They are **not proof of fraud** and do not by themselves "
        "make a system compliant with any regulation.\n"
        "- A real deployment would choose thresholds from the business cost of missed fraud versus false alarms, "
        "and would monitor drift over time.")
