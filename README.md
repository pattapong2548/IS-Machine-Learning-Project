# 📦 Stock Status Predictor — Streamlit App

## โครงสร้างไฟล์
```
my_stock_app/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── soft_voting_tuned.pkl   ← วางไฟล์จาก Colab ตรงนี้
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── features.pkl
└── utils/
    └── predictor.py
```

---

## 🚀 รันบนเครื่อง (Local)

```bash
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. วางไฟล์ .pkl ใน models/

# 3. รัน
streamlit run app.py
```

เปิด browser: http://localhost:8501

---

## ☁️ Deploy บน Streamlit Cloud (ฟรี)

### ขั้นตอน:

1. **สร้าง GitHub repo** ใหม่ (public หรือ private ก็ได้)

2. **Push ไฟล์ทั้งหมดขึ้น GitHub**
```
my_stock_app/
├── app.py
├── requirements.txt
├── models/          ← ต้อง push .pkl ขึ้นไปด้วย!
│   ├── soft_voting_tuned.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── features.pkl
└── utils/
    └── predictor.py
```

3. ไปที่ https://share.streamlit.io
   - Sign in ด้วย GitHub
   - กด **"New app"**
   - เลือก repo / branch / ไฟล์ `app.py`
   - กด **"Deploy!"**

4. รอประมาณ 2–3 นาที ได้ URL ฟรี 🎉
   เช่น: `https://your-app.streamlit.app`

---

## ⚠️ หมายเหตุสำคัญ

- ไฟล์ `.pkl` ต้อง train ด้วย scikit-learn และ xgboost version เดียวกับใน `requirements.txt`
- ถ้า version ไม่ตรงจะ error ตอน load model
- แนะนำให้ train โมเดลใหม่บน Colab แล้วใช้ version ตาม `requirements.txt` นี้เลย
