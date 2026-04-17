# 🚲 S.BIKE — Seoul Bike Sharing Demand Predictor
### BACS3013 Data Science Assignment — Streamlit Deployment

---

## 📦 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## 📁 Project Structure
```
seoul_bike_app/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## 🗂️ Pages

| Page | Description |
|------|-------------|
| 📈 Trend Chart | Line chart of actual vs predicted rentals with prediction start date marker. Filter by date range (2017–2026) and view mode (daily/weekly). |
| 🎯 Single Hour Prediction | Enter any weather + time conditions to predict bike demand. Auto-fill from any date 2017–2026. Works for future dates. |
| 📊 Further Analysis | Model comparison (RF, GB, LR), actual vs predicted scatter, correlation heatmap, seasonal patterns, hourly distribution, feature importance. |

---

## 🤖 Models
- **Random Forest Regressor** (default, best performance)
- **Gradient Boosting Regressor**
- **Linear Regression**

Switch between models in the **sidebar**.

---

## 📊 Dataset
- **Source**: Seoul Bike Sharing Demand — UCI ML Repository
- **Period**: December 2017 – December 2026
- **Frequency**: Hourly
- **Features**: Temperature, Humidity, Wind Speed, Visibility, Dew Point, Solar Radiation, Rainfall, Snowfall, Season, Holiday, Functioning Day
- **Target**: Rented Bike Count

---

## 📚 Reference
Sathishkumar V E, Jangwoo Park, and Yongyun Cho.  
*"Using data mining techniques for bike sharing demand prediction in metropolitan city."*  
Computer Communications, Vol.153, pp.353-366, March 2020.
