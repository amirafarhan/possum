"""Streamlit demo: predict a possum's total body length.

Run with:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

import possum_model as pm

st.set_page_config(page_title="Possum Length Predictor", page_icon="🐾", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return pm.load_data()


@st.cache_resource
def get_trained(df: pd.DataFrame) -> dict:
    return pm.train_and_evaluate(df)


try:
    df = get_data()
    trained = get_trained(df)
except Exception as exc:
    st.error(f"Could not load data or train the model: {exc}")
    st.stop()

model = trained["model"]
stats = df[pm.FEATURES].describe()

st.title("🐾 Possum Total Length Predictor")
st.caption(
    f"Linear regression trained on {len(df)} mountain brushtail possums (`possum.csv`). "
    "Enter body measurements to estimate total body length."
)

tab_predict, tab_batch, tab_model, tab_data = st.tabs(
    ["🔮 Predict", "📄 Batch prediction", "📈 Model performance", "🗂️ Dataset"]
)

# ---------------- Single prediction ----------------
with tab_predict:
    with st.form("measurements"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(pm.FEATURES):
            col = cols[i % 3]
            if feat == "site":
                values[feat] = col.selectbox(pm.FEATURE_LABELS[feat], sorted(df["site"].unique()),
                                             help="Sites 1–2: Victoria; 3–7: New South Wales / Queensland")
            else:
                lo, hi, mean = stats.loc["min", feat], stats.loc["max", feat], stats.loc["mean", feat]
                step = 1.0 if feat == "age" else 0.1
                values[feat] = col.number_input(
                    pm.FEATURE_LABELS[feat], min_value=0.0, max_value=float(hi) * 2,
                    value=float(round(mean, 1)), step=step, help=f"Training range: {lo:g} – {hi:g}",
                )
        submitted = st.form_submit_button("Predict total length", type="primary")

    if submitted:
        out_of_range = [
            pm.FEATURE_LABELS[f] for f in pm.FEATURES
            if f != "site" and not (stats.loc["min", f] <= values[f] <= stats.loc["max", f])
        ]
        if any(values[f] <= 0 for f in pm.FEATURES if f != "site"):
            st.error("All measurements must be greater than zero.")
        else:
            if out_of_range:
                st.warning("Outside the training data range (prediction is an extrapolation): "
                           + ", ".join(out_of_range))
            pred = float(pm.predict(model, values)[0])
            mae = trained["test"]["MAE"]
            c1, c2 = st.columns([1, 2])
            c1.metric("Predicted total length", f"{pred:.1f} cm")
            c1.caption(f"Typical error on held-out data: ±{mae:.1f} cm (MAE)")
            hist = alt.Chart(df).mark_bar(opacity=0.7).encode(
                x=alt.X("totlngth:Q", bin=alt.Bin(maxbins=20), title="Total length (cm)"),
                y=alt.Y("count()", title="Possums"),
            )
            rule = alt.Chart(pd.DataFrame({"x": [pred]})).mark_rule(color="#e4572e", size=3).encode(x="x:Q")
            c2.altair_chart((hist + rule).properties(height=260, title="Prediction vs. dataset distribution"),
                            width="stretch")

# ---------------- Batch prediction ----------------
with tab_batch:
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{f}`" for f in pm.FEATURES))
    st.download_button("Download template (5 rows from the dataset)",
                       df[pm.FEATURES].head().to_csv(index=False), "possum_template.csv", "text/csv")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            result = batch.copy()
            result["predicted_totlngth"] = pm.predict(model, batch).round(2)
            st.success(f"Predicted {len(result)} rows.")
            st.dataframe(result, hide_index=True, width="stretch")
            st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")

# ---------------- Model performance ----------------
with tab_model:
    m = pd.DataFrame({"Train": trained["train"], "Test": trained["test"]}).T
    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{trained['test']['R2']:.3f}")
    c2.metric("Test MAE", f"{trained['test']['MAE']:.2f} cm")
    c3.metric("5-fold CV R²", f"{trained['cv_r2_mean']:.3f} ± {trained['cv_r2_std']:.3f}")
    st.dataframe(m.style.format("{:.3f}"), width="stretch")
    st.caption("The test set has only 21 rows, so its score depends heavily on which possums land in it; "
               "the shuffled 5-fold cross-validated R² is the more reliable estimate.")

    left, right = st.columns(2)
    preds = trained["test_predictions"]
    lo, hi = preds.min().min() - 1, preds.max().max() + 1
    scatter = alt.Chart(preds).mark_circle(size=70).encode(
        x=alt.X("actual:Q", scale=alt.Scale(domain=[lo, hi]), title="Actual (cm)"),
        y=alt.Y("predicted:Q", scale=alt.Scale(domain=[lo, hi]), title="Predicted (cm)"),
        tooltip=["actual", alt.Tooltip("predicted", format=".1f")],
    )
    diag = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="v", y="v")
    left.altair_chart((scatter + diag).properties(title="Actual vs. predicted (test set)", height=320),
                      width="stretch")

    coef = pm.coefficients(model).rename(index=pm.FEATURE_LABELS).reset_index()
    coef.columns = ["feature", "coefficient"]
    right.altair_chart(
        alt.Chart(coef).mark_bar().encode(
            x=alt.X("coefficient:Q", title="cm change per unit"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition("datum.coefficient > 0", alt.value("#4c78a8"), alt.value("#e4572e")),
        ).properties(title="Linear regression coefficients", height=320),
        width="stretch",
    )

# ---------------- Dataset ----------------
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Features used", len(pm.FEATURES))
    c3.metric("Missing values", int(df.isna().sum().sum()))
    corr = df[pm.FEATURES].corrwith(df[pm.TARGET]).rename(index=pm.FEATURE_LABELS).reset_index()
    corr.columns = ["feature", "correlation"]
    st.altair_chart(
        alt.Chart(corr).mark_bar().encode(
            x=alt.X("correlation:Q", scale=alt.Scale(domain=[-1, 1]), title="Correlation with total length"),
            y=alt.Y("feature:N", sort="-x", title=None),
        ).properties(height=300),
        width="stretch",
    )
    st.dataframe(df, hide_index=True, width="stretch")
