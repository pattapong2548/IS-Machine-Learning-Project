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

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e8f0; }
[data-testid="stSidebar"] { background: #111118; border-right: 1px solid #1e1e2e; }
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
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
    font-size: 0.9rem; color: #6b6b8a;
    letter-spacing: 0.05em; text-transform: uppercase; font-weight: 300;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 700; color: #a78bfa;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 0.5rem; margin-bottom: 1.2rem; margin-top: 1.5rem;
}
.metric-card {
    background: #13131f; border: 1px solid #1e1e2e;
    border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #6b6b8a; margin-bottom: 0.3rem;
}
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #e8e8f0; }
.result-in {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #10b981; border-radius: 20px; padding: 2rem 2.4rem; text-align: center;
}
.result-out {
    background: linear-gradient(135deg, #2d0a0a 0%, #4a0d0d 100%);
    border: 1px solid #ef4444; border-radius: 20px; padding: 2rem 2.4rem; text-align: center;
}
.result-label { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin: 0.4rem 0; }
.result-emoji { font-size: 2.5rem; }
.result-prob { font-size: 0.9rem; opacity: 0.7; margin-top: 0.5rem; }
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 12px;
    padding: 0.65rem 2rem; font-family: 'Syne', sans-serif;
    font-weight: 700; font-size: 0.95rem; width: 100%;
}
hr { border-color: #1e1e2e !important; }
label { color: #a0a0b8 !important; font-size: 0.85rem !important; }
.coming-soon {
    background: #13131f; border: 1px dashed #3a3a5a;
    border-radius: 20px; padding: 5rem 2rem; text-align: center;
}
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
CATEGORIES = ["ELECTRONICS", "CLOTHING", "FOOD", "FURNITURE"]


# ─── Predict Function ────────────────────────────────────────────────────────
def predict(category, price, rating, discount):
    row = pd.DataFrame([{"Category": category, "Price": price,
                          "Rating": rating, "Discount": discount}])
    row["Category_enc"]     = le.transform(row["Category"].astype(str))
    row["Price_log"]        = np.log1p(row["Price"])
    row["Price_per_Rating"] = row["Price"] / (row["Rating"] + 0.01)
    row["Discount_flag"]    = (row["Discount"] > 0).astype(int)
    X = scaler.transform(row[features])
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return int(pred), float(proba[1]), float(proba[0])


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Stock Predictor")
    st.markdown("---")
    if not model_loaded:
        st.error("⚠️ ไม่พบไฟล์โมเดลใน `models/`")
        st.stop()
    else:
        st.success("✅ ML Model พร้อมใช้งาน")
    st.markdown("---")
    st.markdown("### 🗺️ เมนู")
    page = st.radio("", options=[
        "📖 ML — อธิบายโมเดล",
        "🔮 ML — ทดสอบโมเดล",
        "📖 Neural Network — อธิบายโมเดล",
        "🤖 Neural Network — ทดสอบโมเดล",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p style="color:#3a3a5a;font-size:0.75rem;text-align:center">Voting Ensemble · RF · XGB · LR · KNN</p>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 1 — ML MODEL INFO
# ════════════════════════════════════════════════════════════════════
if page == "📖 ML — อธิบายโมเดล":

    st.markdown('<div class="hero-title">📖 Machine Learning Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Voting Ensemble · อธิบายแนวทางการพัฒนา</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">1. Dataset & ที่มาของข้อมูล</div>', unsafe_allow_html=True)
    st.markdown("""
**ที่มา:** สร้างโดย Python โดยจำลองข้อมูลร้านค้าออนไลน์ (Synthetic Data with Pattern)

**จำนวน:** 4,000 rows &nbsp;|&nbsp; **Target:** Stock Status (In Stock / Out of Stock)

| Feature | คำอธิบาย | ประเภท |
|---------|----------|--------|
| Category | ประเภทสินค้า (Electronics, Clothing, Food, Furniture) | Categorical |
| Price | ราคาสินค้า (฿50 – ฿10,000) | Numeric |
| Rating | คะแนนรีวิว (1.0 – 5.0) | Numeric |
| Discount | ส่วนลด (0 – 100%) | Numeric |
| **Stock** | **In Stock / Out of Stock** | **Target** |

**Pattern ของข้อมูล:**

| Category | ราคา | แนวโน้ม Stock |
|----------|------|--------------|
| Electronics | สูง (5,000–10,000) | Out of Stock บ่อย |
| Clothing | กลาง (500–3,000) | In Stock บ่อย |
| Food | ต่ำ (50–800) | In Stock เกือบตลอด |
| Furniture | สูงมาก (3,000–10,000) | Out of Stock บ่อย |

**Missing Values:** Category 5% | Price 4% | Rating 6% | Discount 3%
    """)

    st.markdown('<div class="section-header">2. การเตรียมข้อมูล (Data Preparation)</div>', unsafe_allow_html=True)
    st.markdown("""
| ขั้นตอน | วิธีการ | เหตุผล |
|---------|--------|--------|
| Drop empty rows | ลบ rows ที่ว่างทั้งหมด | ข้อมูลไม่มีประโยชน์ |
| Category missing | เติมด้วย Mode | ค่าที่พบบ่อยที่สุดเหมาะสมที่สุด |
| Price missing | เติมด้วย Median แยกตาม Category | Median ทนต่อ outlier ได้ดี |
| Rating missing | เติมด้วย Mean แยกตาม Category | Rating กระจายสม่ำเสมอ |
| Discount missing | เติมด้วย 0 | ถ้าไม่มีข้อมูล = ไม่มี discount |
| Outliers | Winsorisation (IQR method) | ลด noise โดยไม่ drop ข้อมูล |
| Duplicates | ลบ rows ซ้ำ | ป้องกัน data leakage |
| Encoding | LabelEncoder สำหรับ Category | แปลง string → number |
| Scaling | StandardScaler | ทำให้ scale เท่ากันทุก feature |
| Split | Train 60% / Val 20% / Test 20% | 3-way split เพื่อ monitor overfit |
    """)

    st.markdown('<div class="section-header">3. Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown("""
| Feature ใหม่ | สูตร | เหตุผล |
|-------------|------|--------|
| Price_log | log(1 + Price) | ลด skewness ของราคา |
| Price_per_Rating | Price ÷ (Rating + 0.01) | วัดความคุ้มค่าของสินค้า |
| Discount_flag | 1 ถ้า Discount > 0, อื่นๆ = 0 | มี/ไม่มี discount เป็น signal สำคัญ |
    """)

    st.markdown('<div class="section-header">4. ทฤษฎี Voting Ensemble</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**🗳️ Voting Ensemble คืออะไร?**

Ensemble Learning คือการรวมหลายโมเดลเข้าด้วยกัน
เพื่อให้ได้ผลลัพธ์ที่ดีกว่าโมเดลเดี่ยว

**Soft Voting** — เฉลี่ย probability จากทุกโมเดล
แล้วเลือก class ที่มี probability สูงสุด

```
ข้อมูลใหม่
    │
    ├── LR  → prob_in = 0.3
    ├── RF  → prob_in = 0.8
    ├── XGB → prob_in = 0.75
    └── KNN → prob_in = 0.6
         │
    เฉลี่ย = 0.6125 > 0.5
         │
    → In Stock ✅
```
        """)
    with col2:
        st.markdown("""
**🤖 Base Models ที่ใช้**

| Model | ประเภท | จุดเด่น |
|-------|--------|--------|
| Logistic Regression | Linear | เร็ว, interpretable |
| Random Forest | Bagging | robust, handles noise |
| XGBoost | Boosting | แม่นยำสูง |
| KNN | Instance-based | ไม่ assume distribution |

**ทำไมต้องรวม 4 ตัว?**

จุดอ่อนของแต่ละโมเดลจะถูกชดเชยโดยโมเดลอื่น
ทำให้ผลลัพธ์รวมดีกว่าโมเดลเดี่ยวทุกตัว
        """)

    st.markdown('<div class="section-header">5. การป้องกัน Overfitting</div>', unsafe_allow_html=True)
    st.markdown("""
| Technique | ใช้กับ | ผลลัพธ์ |
|-----------|--------|--------|
| Optuna Hyperparameter Tuning (40 trials) | ทุกโมเดล | หา params ที่ generalise ดี |
| max_depth จำกัด | RF, XGBoost | ป้องกัน tree ลึกเกินไป |
| min_samples_leaf | Random Forest | ป้องกัน overfit ของ leaf node |
| subsample + colsample_bytree | XGBoost | สุ่ม row/feature ทุก iteration |
| reg_alpha (L1) + reg_lambda (L2) | XGBoost | Regularization โดยตรง |
| C regularization | Logistic Regression | ควบคุม model complexity |
| 5-Fold Stratified CV | ทุกโมเดล | ประเมินผลอย่างเป็นธรรม |
| Train Gap Monitor | ทุกโมเดล | ตรวจจับ overfit อัตโนมัติ (target < 2%) |
    """)

    st.markdown('<div class="section-header">6. แหล่งอ้างอิง</div>', unsafe_allow_html=True)
    st.markdown("""
1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of KDD '16.
3. Akiba, T., et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD '19.
4. Scikit-learn Documentation — https://scikit-learn.org
5. XGBoost Documentation — https://xgboost.readthedocs.io
6. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
    """)


# ════════════════════════════════════════════════════════════════════
# PAGE 2 — ML PREDICTION
# ════════════════════════════════════════════════════════════════════
elif page == "🔮 ML — ทดสอบโมเดล":

    st.markdown('<div class="hero-title">🔮 ทดสอบ ML Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Voting Ensemble · ทำนาย Stock Status</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">กรอกข้อมูลสินค้า</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", options=CATEGORIES)
        price    = st.slider("Price (฿)", min_value=50, max_value=10000, value=3000, step=50)
    with col2:
        rating   = st.slider("Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
        discount = st.slider("Discount (%)", min_value=0, max_value=100, value=10, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict Stock Status", use_container_width=True)

    if predict_btn:
        pred, prob_in, prob_out = predict(category, price, rating, discount)
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Category</div><div class="metric-value" style="font-size:1.3rem">{category}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Price</div><div class="metric-value">฿{price:,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Rating</div><div class="metric-value">⭐ {rating}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Discount</div><div class="metric-value">{discount}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
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
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_in * 100,
                title={"text": "In Stock Probability (%)", "font": {"color": "#a78bfa", "size": 14}},
                number={"font": {"color": "#e8e8f0", "size": 36}, "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#3a3a5a", "tickfont": {"color": "#6b6b8a"}},
                    "bar": {"color": "#10b981" if pred == 1 else "#ef4444"},
                    "bgcolor": "#13131f", "bordercolor": "#1e1e2e",
                    "steps": [
                        {"range": [0, 40],   "color": "#2d0a0a"},
                        {"range": [40, 60],  "color": "#1a1a2e"},
                        {"range": [60, 100], "color": "#052e16"},
                    ],
                    "threshold": {"line": {"color": "#a78bfa", "width": 2}, "thickness": 0.75, "value": 50},
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
                yaxis={"range": [0, 115], "gridcolor": "#1e1e2e", "tickfont": {"color": "#6b6b8a"}},
                xaxis={"tickfont": {"color": "#c8c8d8"}},
                margin=dict(t=50, b=10, l=30, r=30), showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#3a3a5a;">
            <div style="font-size:3rem">🔮</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; margin-top:1rem">
                กรอกข้อมูลแล้วกด Predict
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — NEURAL NETWORK INFO (Coming Soon)
# ════════════════════════════════════════════════════════════════════
elif page == "📖 Neural Network — อธิบายโมเดล":

    st.markdown('<div class="hero-title">📖 Neural Network Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep Learning · อธิบายแนวทางการพัฒนา</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="coming-soon">
        <div style="font-size:3rem">🚧</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; margin-top:1rem; color:#6b6b8a">
            Coming Soon
        </div>
        <div style="margin-top:0.5rem; color:#3a3a5a; font-size:0.9rem">
            Neural Network Model กำลังอยู่ในระหว่างการพัฒนา
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — NEURAL NETWORK PREDICTION (Coming Soon)
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Neural Network — ทดสอบโมเดล":

    st.markdown('<div class="hero-title">🤖 ทดสอบ Neural Network</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep Learning · ทำนาย Stock Status</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="coming-soon">
        <div style="font-size:3rem">🚧</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; margin-top:1rem; color:#6b6b8a">
            Coming Soon
        </div>
        <div style="margin-top:0.5rem; color:#3a3a5a; font-size:0.9rem">
            Neural Network Model กำลังอยู่ในระหว่างการพัฒนา
        </div>
    </div>
    """, unsafe_allow_html=True)
