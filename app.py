import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111118;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * {
    color: #c8c8d8 !important;
}

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e0e0ff 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6b6b8a;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #a78bfa55; }
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b6b8a;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #e8e8f0;
}
.metric-delta {
    font-size: 0.8rem;
    margin-top: 0.2rem;
}

/* Result banner */
.result-in {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #10b981;
    border-radius: 20px;
    padding: 2rem 2.4rem;
    text-align: center;
}
.result-out {
    background: linear-gradient(135deg, #2d0a0a 0%, #4a0d0d 100%);
    border: 1px solid #ef4444;
    border-radius: 20px;
    padding: 2rem 2.4rem;
    text-align: center;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0.4rem 0;
}
.result-emoji { font-size: 2.5rem; }
.result-prob {
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 0.5rem;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Input labels */
label { color: #a0a0b8 !important; font-size: 0.85rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    width: 100%;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #13131f;
    border-radius: 12px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6b6b8a;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: #1e1e2e !important;
    color: #e8e8f0 !important;
}

/* Divider */
hr { border-color: #1e1e2e !important; }

/* Plotly charts background */
.js-plotly-plot { border-radius: 16px; overflow: hidden; }

/* Selectbox / slider */
.stSelectbox > div > div,
.stSlider > div { background: #13131f !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load(MODEL_DIR / "soft_voting_tuned.pkl")
        scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
        le       = joblib.load(MODEL_DIR / "label_encoder.pkl")
        features = joblib.load(MODEL_DIR / "features.pkl")
        return model, scaler, le, features, True
    except Exception as e:
        return None, None, None, None, False

model, scaler, le, features, model_loaded = load_artifacts()

CATEGORIES = ["Electronics", "Clothing", "Food", "Furniture"]


# ─── Prediction Function ─────────────────────────────────────────────────────
def predict(category, price, rating, discount):
    row = pd.DataFrame([{
        "Category": category,
        "Price":    price,
        "Rating":   rating,
        "Discount": discount,
    }])
    row["Category_enc"]     = le.transform(row["Category"].astype(str))
    row["Price_log"]        = np.log1p(row["Price"])
    row["Price_per_Rating"] = row["Price"] / (row["Rating"] + 0.01)
    row["Discount_flag"]    = (row["Discount"] > 0).astype(int)

    X     = scaler.transform(row[features])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return int(pred), float(proba[1]), float(proba[0])


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Stock Predictor")
    st.markdown("---")

    if not model_loaded:
        st.error("⚠️ ไม่พบไฟล์โมเดลใน `models/`\n\nกรุณาวางไฟล์ `.pkl` ก่อนรันครับ")
        st.stop()
    else:
        st.success("✅ โมเดลพร้อมใช้งาน")

    st.markdown("---")
    st.markdown('<div class="section-header">🔧 Input Parameters</div>', unsafe_allow_html=True)

    category = st.selectbox("Category", options=CATEGORIES, index=0)
    price    = st.slider("Price (฿)", min_value=100, max_value=10000,
                          value=3000, step=50)
    rating   = st.slider("Rating", min_value=1.0, max_value=5.0,
                          value=3.5, step=0.1)
    discount = st.slider("Discount (%)", min_value=0, max_value=100,
                          value=10, step=1)

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Stock Status", use_container_width=True)

    st.markdown("---")
    st.markdown('<p style="color:#3a3a5a;font-size:0.75rem;text-align:center">Voting Ensemble · 4 Models<br>RF · XGB · LR · KNN</p>',
                unsafe_allow_html=True)


# ─── Main Layout ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Stock Status Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Voting Ensemble · 4-Model AI · Anti-Overfit</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮  Predict", "📊  Dashboard", "ℹ️  Model Info"])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════
with tab1:

    # Input summary row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Category</div><div class="metric-value">{category}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Price</div><div class="metric-value">฿{price:,}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Rating</div><div class="metric-value">⭐ {rating}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Discount</div><div class="metric-value">{discount}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if predict_btn:
        pred, prob_in, prob_out = predict(category, price, rating, discount)

        # Result banner
        rc1, rc2, rc3 = st.columns([1, 2, 1])
        with rc2:
            if pred == 1:
                st.markdown(f"""
                <div class="result-in">
                    <div class="result-emoji">✅</div>
                    <div class="result-label" style="color:#10b981">In Stock</div>
                    <div class="result-prob">ความมั่นใจ {prob_in*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-out">
                    <div class="result-emoji">❌</div>
                    <div class="result-label" style="color:#ef4444">Out of Stock</div>
                    <div class="result-prob">ความมั่นใจ {prob_out*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability gauge
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_in * 100,
                title={"text": "In Stock Probability (%)",
                       "font": {"color": "#a78bfa", "size": 14}},
                number={"font": {"color": "#e8e8f0", "size": 36}, "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#3a3a5a",
                              "tickfont": {"color": "#6b6b8a"}},
                    "bar": {"color": "#10b981" if pred == 1 else "#ef4444"},
                    "bgcolor": "#13131f",
                    "bordercolor": "#1e1e2e",
                    "steps": [
                        {"range": [0, 40],  "color": "#2d0a0a"},
                        {"range": [40, 60], "color": "#1a1a2e"},
                        {"range": [60, 100],"color": "#052e16"},
                    ],
                    "threshold": {"line": {"color": "#a78bfa", "width": 2},
                                  "thickness": 0.75, "value": 50},
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0a0a0f", plot_bgcolor="#0a0a0f",
                font_color="#e8e8f0", height=260, margin=dict(t=50, b=10, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with gc2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=["Out of Stock", "In Stock"],
                y=[prob_out * 100, prob_in * 100],
                marker_color=["#ef4444", "#10b981"],
                text=[f"{prob_out*100:.1f}%", f"{prob_in*100:.1f}%"],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title={"text": "Class Probabilities", "font": {"color": "#a78bfa", "size": 14}},
                paper_bgcolor="#0a0a0f", plot_bgcolor="#13131f",
                font_color="#e8e8f0", height=260,
                yaxis={"range": [0, 115], "gridcolor": "#1e1e2e",
                        "tickfont": {"color": "#6b6b8a"}},
                xaxis={"tickfont": {"color": "#c8c8d8"}},
                margin=dict(t=50, b=10, l=30, r=30),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#3a3a5a;">
            <div style="font-size:3rem">🔮</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; margin-top:1rem">
                ปรับค่าด้านซ้าย แล้วกด Predict
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD (Batch Simulation)
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📊 Batch Simulation Dashboard</div>', unsafe_allow_html=True)
    st.caption("จำลองการทำนาย 200 รายการสุ่ม เพื่อดู distribution ของโมเดล")

    if st.button("▶  Run Simulation (200 samples)", key="sim_btn"):
        with st.spinner("กำลังจำลอง..."):
            np.random.seed(42)
            n = 200
            sim_df = pd.DataFrame({
                "Category": np.random.choice(["A","B","C","D"], n),
                "Price":    np.random.uniform(400, 9500, n).round(0),
                "Rating":   np.random.uniform(1.0, 5.0, n).round(2),
                "Discount": np.random.uniform(0, 50, n).round(0),
            })

            sim_df["Category_enc"]     = le.transform(sim_df["Category"].astype(str))
            sim_df["Price_log"]        = np.log1p(sim_df["Price"])
            sim_df["Price_per_Rating"] = sim_df["Price"] / (sim_df["Rating"] + 0.01)
            sim_df["Discount_flag"]    = (sim_df["Discount"] > 0).astype(int)

            X_sim      = scaler.transform(sim_df[features])
            preds_sim  = model.predict(X_sim)
            probas_sim = model.predict_proba(X_sim)[:, 1]

            sim_df["Prediction"] = ["In Stock" if p==1 else "Out of Stock" for p in preds_sim]
            sim_df["Confidence"] = (probas_sim * 100).round(1)

        # KPI row
        n_in  = (preds_sim == 1).sum()
        n_out = (preds_sim == 0).sum()
        avg_conf = probas_sim.mean() * 100

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Samples</div><div class="metric-value">{n}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">In Stock</div><div class="metric-value" style="color:#10b981">{n_in}</div><div class="metric-delta" style="color:#10b981">▲ {n_in/n*100:.1f}%</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Out of Stock</div><div class="metric-value" style="color:#ef4444">{n_out}</div><div class="metric-delta" style="color:#ef4444">▼ {n_out/n*100:.1f}%</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Confidence</div><div class="metric-value">{avg_conf:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Charts row
        ch1, ch2 = st.columns(2)

        with ch1:
            counts = sim_df["Prediction"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.55,
                marker_colors=["#10b981", "#ef4444"],
                marker_line=dict(color="#0a0a0f", width=3),
                textfont={"color": "#e8e8f0", "size": 13},
            ))
            fig_pie.update_layout(
                title={"text": "Prediction Distribution", "font": {"color":"#a78bfa","size":14}},
                paper_bgcolor="#0a0a0f", font_color="#e8e8f0",
                legend={"font": {"color": "#c8c8d8"}},
                height=320, margin=dict(t=50,b=10,l=10,r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            fig_conf = px.histogram(
                sim_df, x="Confidence", color="Prediction",
                nbins=25, barmode="overlay", opacity=0.7,
                color_discrete_map={"In Stock":"#10b981","Out of Stock":"#ef4444"},
            )
            fig_conf.update_layout(
                title={"text": "Confidence Distribution", "font":{"color":"#a78bfa","size":14}},
                paper_bgcolor="#0a0a0f", plot_bgcolor="#13131f",
                font_color="#e8e8f0", height=320,
                xaxis={"gridcolor":"#1e1e2e","tickfont":{"color":"#6b6b8a"},"title":"Confidence (%)"},
                yaxis={"gridcolor":"#1e1e2e","tickfont":{"color":"#6b6b8a"}},
                legend={"font":{"color":"#c8c8d8"}},
                margin=dict(t=50,b=40,l=10,r=10)
            )
            st.plotly_chart(fig_conf, use_container_width=True)

        # Category breakdown
        cat_counts = sim_df.groupby(["Category","Prediction"]).size().unstack(fill_value=0)
        fig_cat = go.Figure()
        for col, color in [("In Stock","#10b981"),("Out of Stock","#ef4444")]:
            if col in cat_counts.columns:
                fig_cat.add_trace(go.Bar(
                    name=col, x=cat_counts.index, y=cat_counts[col],
                    marker_color=color+"99", marker_line_color=color, marker_line_width=1.5
                ))
        fig_cat.update_layout(
            title={"text": "Prediction by Category", "font":{"color":"#a78bfa","size":14}},
            barmode="group",
            paper_bgcolor="#0a0a0f", plot_bgcolor="#13131f",
            font_color="#e8e8f0", height=300,
            xaxis={"gridcolor":"#1e1e2e","tickfont":{"color":"#c8c8d8"}},
            yaxis={"gridcolor":"#1e1e2e","tickfont":{"color":"#6b6b8a"}},
            legend={"font":{"color":"#c8c8d8"}},
            margin=dict(t=50,b=40,l=10,r=10)
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        # Price vs Confidence scatter
        fig_scatter = px.scatter(
            sim_df, x="Price", y="Confidence", color="Prediction",
            size="Discount", hover_data=["Category","Rating"],
            color_discrete_map={"In Stock":"#10b981","Out of Stock":"#ef4444"},
            opacity=0.6,
        )
        fig_scatter.update_layout(
            title={"text": "Price vs Confidence (size = Discount)", "font":{"color":"#a78bfa","size":14}},
            paper_bgcolor="#0a0a0f", plot_bgcolor="#13131f",
            font_color="#e8e8f0", height=340,
            xaxis={"gridcolor":"#1e1e2e","tickfont":{"color":"#6b6b8a"}},
            yaxis={"gridcolor":"#1e1e2e","tickfont":{"color":"#6b6b8a"}},
            legend={"font":{"color":"#c8c8d8"}},
            margin=dict(t=50,b=40,l=10,r=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">📋 Sample Data (10 rows)</div>', unsafe_allow_html=True)
        display_cols = ["Category","Price","Rating","Discount","Prediction","Confidence"]
        st.dataframe(
            sim_df[display_cols].head(10).style
                .applymap(lambda v: "color:#10b981" if v=="In Stock"
                          else ("color:#ef4444" if v=="Out of Stock" else ""),
                          subset=["Prediction"])
                .format({"Price": "฿{:.0f}", "Confidence": "{:.1f}%"}),
            use_container_width=True,
        )
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#3a3a5a;">
            <div style="font-size:3rem">📊</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; margin-top:1rem">
                กด Run Simulation เพื่อดู Dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">ℹ️ Model Architecture</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        **🗳️ Voting Ensemble (Soft Vote)**

        โมเดลรวม 4 ตัว โหวตผ่าน probability เฉลี่ย:

        | # | Model | ประเภท |
        |---|-------|--------|
        | 1 | Logistic Regression | Linear |
        | 2 | Random Forest | Bagging |
        | 3 | XGBoost | Boosting |
        | 4 | KNN | Instance-based |
        """)

    with col_b:
        st.markdown("""
        **🛡️ Anti-Overfit Techniques**

        | Technique | โมเดล |
        |-----------|-------|
        | L1/L2 Regularization | LR, XGBoost |
        | max_depth จำกัด | RF, XGBoost |
        | subsample + colsample | XGBoost |
        | min_samples_leaf | Random Forest |
        | Optuna Tuning (CV objective) | ทุกตัว |
        | 5-Fold Stratified CV | ทุกตัว |
        """)

    st.markdown("---")
    st.markdown('<div class="section-header">📥 Features ที่ใช้</div>', unsafe_allow_html=True)

    feat_data = {
        "Feature": ["Category_enc","Price","Price_log","Rating",
                    "Discount","Price_per_Rating","Discount_flag"],
        "คำอธิบาย": ["Category A/B/C/D → ตัวเลข",
                      "ราคาสินค้า (฿)",
                      "log(1+Price) — ลด skewness",
                      "คะแนน 1.0–5.0",
                      "ส่วนลด 0–100%",
                      "Price ÷ Rating — ratio feature",
                      "0/1 ว่ามี discount หรือไม่"],
        "ประเภท": ["Encoded","Numeric","Engineered","Numeric",
                    "Numeric","Engineered","Binary"],
    }
    st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📌 หมายเหตุ</div>', unsafe_allow_html=True)
    st.info("""
    Dataset นี้เป็น **synthetic random data** ดังนั้น accuracy จริงอยู่ที่ประมาณ **50–65%**
    ซึ่งถือว่า honest และ generalise ดี — ดีกว่าได้ 95% แต่ overfit ครับ
    """)
